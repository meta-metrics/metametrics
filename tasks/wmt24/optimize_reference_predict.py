from bayes_opt import BayesianOptimization
from scipy import stats
import csv
import json

import pandas as pd

def run(df):
    metric_scores = {}
    human_scores = df['human_score'].to_list()
    df = df.drop(columns=['lp', 'domain', 'year', 'id', 'human_score'])
    df = df.drop(columns=['GEMBA_score', 'metricx-23-qe-large-v2p0_reference_free', 'metricx-23-qe-xl-v2p0_reference_free', 'metricx-23-qe-xxl-v2p0_reference_free', 'wmt22-cometkiwi-da_reference_free', 'wmt23-cometkiwi-da-xl_reference_free'])
    
    metrics_names = list(df.columns)
    for metric_name in metrics_names:
        metric_scores[metric_name] = []
    
    for metric_name in metrics_names:
        if metric_name != "human_score":
            metric_scores[metric_name] = df[metric_name].to_list()

    metric_weights = {
      "bertscore_f1": 0.0,
      "bertscore_precision": 0.0,
      "bleu": 0.0,
      "bleurt": 1.0,
      "chrf": 0.0,
      "metricx-23-large-v2p0": 0.2693398486515744,
      "metricx-23-xl-v2p0": 0.0,
      "metricx-23-xxl-v2p0": 1.0,
      "wmt22-comet-da": 1.0,
      "xcomet-xl": 0.0,
      "yisi": 0.0
    }
    
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

df1 = pd.read_csv('all/wmt-sqm-human-evaluation_score_final_scaled.csv')
df2 = pd.read_csv('all/wmt-mqm-human-evaluation_score_final_scaled.csv')
df3 = pd.read_csv('all/wmt-da-human-evaluation_score_final_scaled.csv')
combined_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

res = {"sqm": run(df1), "mqm": run(df2), "da": run(df3)}
print(res)