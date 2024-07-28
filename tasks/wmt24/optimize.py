from bayes_opt import BayesianOptimization
from scipy import stats

import pandas as pd

human_scores = []
metric_scores = {}

df = pd.read_csv('wmt-sqm-human-evaluation_score.csv')
df = df.drop(columns=["id"])
human_scores = df['human_score'].to_list()
df = df.drop(columns=["human_score"])
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
    init_points=2,
    n_iter=40,
)

print(optimizer.max)