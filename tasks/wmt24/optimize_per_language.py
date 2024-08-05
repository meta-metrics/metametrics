from bayes_opt import BayesianOptimization
from scipy import stats
import csv
import json

import pandas as pd

def run(df, language):
    metric_scores = {}
    df = df[df["lp"]==language]
    human_scores = df['human_score'].to_list()
    df = df.drop(columns=['lp', 'domain', 'year', 'id', 'human_score'])
    df = df.drop(columns=['GEMBA_score', 'metricx-23-qe-large-v2p0_reference_free', 'metricx-23-qe-xl-v2p0_reference_free', 'metricx-23-qe-xxl-v2p0_reference_free', 'wmt22-cometkiwi-da_reference_free', 'wmt23-cometkiwi-da-xl_reference_free'])
    
    metrics_names = list(df.columns)
    for metric_name in metrics_names:
        metric_scores[metric_name] = []
    
    for metric_name in metrics_names:
        if metric_name != "human_score":
            metric_scores[metric_name] = df[metric_name].to_list()
            kendall_score = stats.kendalltau(human_scores, metric_scores[metric_name])
            print(metric_name, kendall_score.correlation)
    
    print("metrics names:", metrics_names)
    
    # Bounded region of parameter space
    pbounds = {}
    for metric_name in metrics_names:
        pbounds[metric_name] = (0, 1)
    
    def black_box_function(**metric_weights):
        """Function with unknown internals we wish to maximize.
    
        This is just serving as an example, for all intents and
        purposes think of the internals of this function, i.e.: the process
        which generates its output values, as unknown.
        """
        final_metric_scores = []
        print("metric_weights:", metric_weights)
        
        count = 0
        for i in range(len(human_scores)): # data
            score = 0
            for metric_name in metric_scores: # metrics
                score += metric_scores[metric_name][i] * metric_weights[metric_name]
            final_metric_scores.append(score)
            count += 1
    
        # calculate kendall
        kendall_score = stats.kendalltau(human_scores, final_metric_scores)
        return kendall_score.correlation
    
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=1,
    )
    
    optimizer.maximize(
        init_points=5,
        n_iter=100,
    )
    
    return optimizer.max

language = "en-de"

df1 = pd.read_csv('all/wmt-sqm-human-evaluation_score_final_scaled.csv')
df2 = pd.read_csv('all/wmt-mqm-human-evaluation_score_final_scaled.csv')
df3 = pd.read_csv('all/wmt-da-human-evaluation_score_final_scaled.csv')
combined_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

res = {"sqm": run(df1,language), "mqm": run(df2,language), "da": run(df3,language), "combined": run(combined_df,language)}

with open(f"results_{language}_with_references.json", "w+") as f:
    json.dump(res, f, indent = 2)