from bayes_opt import BayesianOptimization
from scipy import stats

import pandas as pd

human_scores = []
metric_scores = {}

df = pd.read_csv('data.csv')
df.drop(columns=["id"])

metrics_names = list(df.columns)
for metric_name in metrics_names:
    if metric_name != "human_score":
        metric_scores[metric_name] = True

for index, row in df.iterrows():
    human_scores.append(row['human_score'])
    for metric_name in metrics_names:
        metric_scores[metric_name].append(row[metric_name])

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
    
    count = 0
    for i in range(len(metric_weights)):
        for j in range(len(dataset:
            if i == 0:
                human_scores.append(data["human_score"])
                final_metric_scores.append(data["metric_score"] * metric_weights[i])
            else:
                final_metric_scores[j] += data["metric_score"] * metric_weights[i]
        count += 1

    # calculate kendall
    kendall_score = stats.kendalltau(human_scores, final_metric_scores)
    return kendall_score

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=40,
)