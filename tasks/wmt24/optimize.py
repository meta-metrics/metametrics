from bayes_opt import BayesianOptimization

import pandas as pd

df = pd.read_csv('data.csv')

model_names = list(df.columns)


# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)