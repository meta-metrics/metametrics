import numpy as np
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from scipy import stats
from multiprocessing import Process, Queue
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import logging

logging.basicConfig(level=logging.INFO)

MAX_WORKERS = 16

def prepare_data(ori_df, split, group_system, drop_columns):
    """Prepare train, validation, and test datasets."""
    df = ori_df.copy()
    train_df, test_df = train_test_split(df, test_size=split, random_state=1, stratify=df[group_system])
    train_val_df, val_df = train_test_split(df, test_size=(0.1 / (1 - split)), random_state=1, stratify=df[group_system])
    train_val_df = train_val_df.drop(columns=drop_columns)
    train_df = train_df.drop(columns=drop_columns)
    return train_df, train_val_df, val_df, test_df

def calculate_initial_metric_points(train_df, human_scor, human_evals_list):
    """Calculate initial metric points based on Kendall's tau correlation."""
    metric_scores = {}
    human_scores = train_df[human_scor].to_list()
    metrics_names = list(train_df.drop(columns=human_evals_list).columns)
    initial_metric_point = np.empty((1, len(metrics_names)))

    for i, metric_name in enumerate(metrics_names):
        metric_scores[metric_name] = train_df[metric_name].to_list()
        initial_metric_point.T[i] = abs(stats.kendalltau(human_scores, metric_scores[metric_name]).correlation)
    
    return metric_scores, metrics_names, human_scores, initial_metric_point

def black_box_function(metric_scores, human_scores, metric_weights, objective):
    """Calculate the objective function score."""
    final_metric_scores = [
        sum(metric_scores[metric_name][i] * metric_weights[metric_name] for metric_name in metric_scores)
        for i in range(len(human_scores))
    ]
    if objective == "kendall":
        return stats.kendalltau(human_scores, final_metric_scores).correlation
    elif objective == "pearson":
        return stats.pearsonr(human_scores, final_metric_scores).statistic
    elif objective == "kendall_pearson":
        kendall_score = stats.kendalltau(human_scores, final_metric_scores).correlation
        pearson_score = stats.pearsonr(human_scores, final_metric_scores).statistic
        return kendall_score + pearson_score

def run_bayesian_optimization(metric_scores, metrics_names, human_scores, objective, seed, init_points, n_iter, initial_metric_point):
    """Run Bayesian optimization with given parameters."""
    def wrapped_black_box_function(**metric_weights):
        return black_box_function(metric_scores, human_scores, metric_weights, objective)
    
    pbounds = {metric_name: (0, 1) for metric_name in metrics_names}
    optimizer = BayesianOptimization(f=wrapped_black_box_function, pbounds=pbounds, random_state=seed)
    optimizer.maximize(init_points=init_points, n_iter=n_iter, initial_guess=initial_metric_point)
    
    return optimizer

def optimize_with_seed(seed, metrics, human_evals_list, train_df, val_df, human_scor, objective, init_points, n_iter):
    """Optimize metrics weights based on given seed."""
    logging.info(f"Running with seed {seed}")
    metric_scores, metrics_names, human_scores, initial_metric_point = calculate_initial_metric_points(train_df, human_scor, human_evals_list)
    optimizer = run_bayesian_optimization(metric_scores, metrics_names, human_scores, objective, seed, init_points, n_iter, initial_metric_point)
    
    # Evaluate on validation data
    copy_val_df = val_df.copy()
    copy_val_df['final_score'] = copy_val_df.apply(
        lambda row: sum(row[col] * optimizer.max['params'].get(col, 0) for col in metrics), axis=1)
    kt = stats.kendalltau(copy_val_df["final_score"], copy_val_df[human_scor], nan_policy='omit')
    
    return (seed, kt.correlation)

def get_weight_optimization(ori_df, metrics, drop_columns, splits_test, human_evals_list, group_system, objective="kendall", init_points=5, n_iter=100, n_seeds=100):
    # Preprocess metric values in the original dataframe
    ori_df["chrf"] = ori_df["chrf"] / 100
    ori_df["bleu"] = ori_df["bleu"] / 100
    if "bartscore" in metrics:
        ori_df["bartscore"] = np.exp(ori_df["bartscore"])
    else:
        ori_df["bartscore_mean"] = np.exp(ori_df["bartscore_mean"])
        ori_df["bartscore_max"] = np.exp(ori_df["bartscore_max"])

    final_result_opt = []
    for split in splits_test:
        train_df, train_val_df, val_df, test_df = prepare_data(ori_df, split, group_system, drop_columns)
        
        for human_scor in human_evals_list:
            results = []

            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(optimize_with_seed, seed, metrics, human_evals_list, train_val_df, val_df, human_scor, objective, init_points, n_iter): seed 
                           for seed in range(n_seeds)}

                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logging.error(f"An error occurred with seed {seed}: {e}")

            # Determine the best result and seed
            best_seed_result = max(results, key=lambda x: x[1])

            logging.info(f"Best seed for {human_scor}: {best_seed_result[0]} with score {best_seed_result[1]}")
            
            ## Now begin actual training
            metric_scores, metrics_names, human_scores, initial_metric_point = calculate_initial_metric_points(train_df, human_scor, human_evals_list)
            optimizer = run_bayesian_optimization(metric_scores, metrics_names, human_scores, objective, best_seed_result[0], init_points, n_iter, initial_metric_point)
            final_result_opt.append(optimizer.max)
            
    return final_result_opt

def get_correlation_ours(any_df, human_evals_list, final_he_list, group_system, folder_name, csv_name):
    correlation_table = pd.DataFrame(index=["sys", "seg"], columns=(human_evals_list + ["avg"]))
    avg_sys_correlation = []
    for ex, fin in zip(human_evals_list, final_he_list):
        means_df = any_df.copy(True).groupby(group_system).agg({ex: 'mean', fin: 'mean'}).reset_index()
        # Compute Kendall tau correlation coefficient
        kt = stats.kendalltau(means_df[ex], means_df[fin], nan_policy='omit')
        avg_sys_correlation.append(kt.correlation)
    avg_sys_correlation.append(np.mean(np.array(avg_sys_correlation)))
    
    avg_seg_correlation = []
    for ex, fin in zip(human_evals_list, final_he_list):
        means_df = any_df.copy(True)
        # Compute Kendall tau correlation coefficient
        kt = stats.kendalltau(means_df[ex], means_df[fin], nan_policy='omit')
        avg_seg_correlation.append(kt.correlation)
    avg_seg_correlation.append(np.mean(np.array(avg_seg_correlation)))
    
    correlation_table.loc["sys"] = avg_sys_correlation
    correlation_table.loc["seg"] = avg_seg_correlation
    
    correlation_table.to_csv(f"{folder_name}/{csv_name}_ours.csv")
    
def get_correlation_test_other_metrics(test_df, human_evals_list, metrics, group_system, folder_name, csv_name):
    correlation_table = pd.DataFrame(index=metrics, columns=human_evals_list)

    # Iterate over each combination of human_eval and metric to calculate Kendall tau
    for human_eval in human_evals_list:
        for metric in metrics:
            means_df = test_df.copy(True).groupby(group_system).agg({human_eval: 'mean', metric: 'mean'}).reset_index()
            # Compute Kendall tau correlation coefficient
            kt = stats.kendalltau(means_df[human_eval], means_df[metric], nan_policy='omit')
            correlation_table.loc[metric, human_eval] = kt.correlation
            
    correlation_table.to_csv(f"{folder_name}/{csv_name}_sys.csv")

    correlation_table = pd.DataFrame(index=metrics, columns=human_evals_list)

    # Iterate over each combination of human_eval and metric to calculate Kendall tau
    for human_eval in human_evals_list:
        for metric in metrics:
            # Compute Kendall tau correlation coefficient
            kt = stats.kendalltau(test_df[human_eval], test_df[metric], nan_policy='omit')
            correlation_table.loc[metric, human_eval] = kt.correlation
            
    correlation_table.to_csv(f"{folder_name}/{csv_name}_seg.csv")
            
            
