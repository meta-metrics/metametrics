import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
import pandas as pd
import json
import logging

# For models
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO)

MAX_WORKERS = 16

def prepare_data(ori_df, split, group_system, drop_columns):
    """Prepare train, validation, and test datasets."""
    df = ori_df.copy()
    train_df, test_df = train_test_split(df, test_size=split, random_state=1, stratify=df[group_system])
    return train_df, test_df

# Define the custom Kendall scorer
def kendall_scorer(y_true, y_pred):
    # Calculate Kendall's Tau correlation
    score = stats.kendalltau(y_true, y_pred).correlation
    return score

def optimize_xgb(xgb_json, xgb_json2, train_df, human_scor, metrics, group_system):
    X_train = train_df[metrics]
    Y_train = train_df[human_scor].to_list()
    
    with open(xgb_json, 'r') as file:
        json_content = json.load(file)

    fixed_param = json_content.get("fixed_param", {})

    # Initialize BayesSearchCV through parameter space
    param_space = {}
    for param_name, param_info in json_content["param_space"].items():
        param_type = param_info["type"]
        low = param_info["low"]
        high = param_info["high"]
        prior = param_info["prior"] if "prior" in param_info else "uniform"
        if param_type == "Integer":
            param_space[param_name] = Integer(low, high, prior=prior)
        elif param_type == "Real":
            param_space[param_name] = Real(low, high, prior=prior)
        elif param_type == "Categorical":
            step = param_info["step"] if "step" in param_info else 1
            param_space[param_name] = Categorical(list(range(low, high + 1, step)))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    n_iter = json_content["n_iter"]

    # Create the scorer function
    scorer = make_scorer(kendall_scorer, greater_is_better=True)

    # Initialize the XGBoost model
    xgb_model = XGBRegressor(tree_method='auto', objective="reg:absoluteerror", random_state=1, **fixed_param)

    logging.info('Running XGB Bayes Search')
    logging.info(f"Parameter space: {param_space}")
    hpm_search_algorithm = BayesSearchCV(estimator=xgb_model,
                                            search_spaces=param_space,
                                            scoring=scorer,
                                            cv=10,
                                            n_iter=n_iter,
                                            random_state=1,
                                            n_jobs=-1)
    
    

    # Fit the grid search to the data
    hpm_search_algorithm.fit(X_train, Y_train)
    logging.info("Best params:")
    logging.info(hpm_search_algorithm.best_params_)

    # Get the best model
    best_model = hpm_search_algorithm.best_estimator_
    logging.info("Feature importance:")
    logging.info(best_model.feature_importances_)    
    
    with open(xgb_json2, 'r') as file:
        json_content = json.load(file)

    fixed_param = json_content.get("fixed_param", {})

    # Initialize BayesSearchCV through parameter space
    param_space = {}
    for param_name, param_info in json_content["param_space"].items():
        param_type = param_info["type"]
        low = param_info["low"]
        high = param_info["high"]
        prior = param_info["prior"] if "prior" in param_info else "uniform"
        if param_type == "Integer":
            param_space[param_name] = Integer(low, high, prior=prior)
        elif param_type == "Real":
            param_space[param_name] = Real(low, high, prior=prior)
        elif param_type == "Categorical":
            step = param_info["step"] if "step" in param_info else 1
            param_space[param_name] = Categorical(list(range(low, high + 1, step)))
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    n_iter = json_content["n_iter"]
    
    
    
    # new_xgb_model = XGBRegressor(tree_method='auto', objective="reg:absoluteerror", random_state=1, **fixed_param)

    # logging.info('Running XGB Bayes Search')
    # logging.info(f"Parameter space: {param_space}")
    # hpm_search_algorithm = BayesSearchCV(estimator=new_xgb_model,
    #                                         search_spaces=param_space,
    #                                         scoring=scorer,
    #                                         cv=3,
    #                                         n_iter=n_iter,
    #                                         random_state=1,
    #                                         n_jobs=-1)

    # Fit the grid search to the data; train further
    predicted_score = best_model.predict(X_train)
    new_train_df = train_df.copy(True)
    new_train_df['new_columns'] = predicted_score
    ours_y_score = new_train_df.groupby(group_system).agg({'new_columns': 'mean'}).reset_index()['new_columns'].to_numpy().astype(np.float32).reshape(-1, 1)
    sys_y_score = new_train_df.copy(True).groupby(group_system).agg({human_scor: 'mean'}).reset_index()[human_scor].to_numpy().astype(np.float32)

    # hpm_search_algorithm.fit(ours_y_score, sys_y_score)
    # logging.info("Best params:")
    # logging.info(hpm_search_algorithm.best_params_)

    # # Get the best model
    # sys_best_model = hpm_search_algorithm.best_estimator_
    # logging.info("Feature importance:")
    # logging.info(sys_best_model.feature_importances_)
    
    rf_model = RandomForestRegressor(n_estimators=20, random_state=1)

    # Fit the model
    rf_model.fit(ours_y_score, sys_y_score)

    return best_model, rf_model

def get_weight_optimization(ori_df, metrics, drop_columns, splits_test, human_evals_list, group_system, xgb_json_path, xgb_json_path2):
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
        train_df, test_df = prepare_data(ori_df, split, group_system, drop_columns)
        
        for human_scor in human_evals_list:
            final_result_opt.append(optimize_xgb(xgb_json_path, xgb_json_path2, train_df, human_scor, metrics, group_system))
            
    return final_result_opt

def get_correlation_ours_sys(any_df, human_evals_list, sys_scores, group_system, folder_name, csv_name):
    correlation_table = pd.DataFrame(index=["sys", "seg"], columns=(human_evals_list + ["avg"]))
    avg_sys_correlation = []
    for ex, fin in zip(human_evals_list, sys_scores):
        means_df = any_df.copy(True).groupby(group_system).agg({ex: 'mean'}).reset_index()
        # Compute Kendall tau correlation coefficient
        kt = stats.kendalltau(means_df[ex], fin, nan_policy='omit')
        avg_sys_correlation.append(kt.correlation)
    avg_sys_correlation.append(np.mean(np.array(avg_sys_correlation)))
    
    correlation_table.loc["sys"] = avg_sys_correlation
    
    correlation_table.to_csv(f"{folder_name}/{csv_name}_ours_sys.csv")

def get_correlation_ours_seg(any_df, human_evals_list, final_he_list, group_system, folder_name, csv_name):
    correlation_table = pd.DataFrame(index=["sys", "seg"], columns=(human_evals_list + ["avg"]))
    avg_seg_correlation = []
    for ex, fin in zip(human_evals_list, final_he_list):
        means_df = any_df.copy(True)
        # Compute Kendall tau correlation coefficient
        kt = stats.kendalltau(means_df[ex], means_df[fin], nan_policy='omit')
        avg_seg_correlation.append(kt.correlation)
    avg_seg_correlation.append(np.mean(np.array(avg_seg_correlation)))
    correlation_table.loc["seg"] = avg_seg_correlation
    
    correlation_table.to_csv(f"{folder_name}/{csv_name}_ours_seg.csv")
    
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