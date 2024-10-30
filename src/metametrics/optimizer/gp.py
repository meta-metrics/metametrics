
import json
from bayes_opt import BayesianOptimization
import numpy as np

from metametrics.optimizer.base_optimizer import BaseOptimizer
from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class GaussianProcessOptimizer(BaseOptimizer):
    def __init__(self, objective_fn, init_points=5, n_iter=100, seed=1):
        self.objective_fn = objective_fn
        self.init_points = init_points
        self.n_iter = n_iter
        self.seed = seed
        self.need_calibrate = True
        self.optimizer = {}
        
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            json_content = json.load(file)
            
        self.init_points = json_content.get("init_points", 5)
        self.n_iter = json_content.get("n_iter", 100)
        self.seed = json_content.get("seed", 1)
        self.need_calibrate = True
        self.optimizer = {}
        
        # Register objective function based on user-provided string
        objective_fn_str = json_content.get("objective_fn")
        if objective_fn_str in objective_fn_map:
            self.objective_fn = objective_fn_map[objective_fn_str]
        else:
            raise ValueError(f"Objective function '{objective_fn_str}' is not recognized.")

    def black_box_function(self, metric_array, target_scores, metric_weights):
        """Calculate the objective function score."""
        # Compute final_metric_scores as a dot product of the metrics and weights
        metric_names = metric_weights.keys()
        weights_array = np.array([metric_weights[metric_name] for metric_name in metric_names])

        # Check if final_metric_scores are all zero
        if np.all(weights_array == 0):
            metric_weights = {
                metric_name: np.random.uniform(low=0.01, high=1) for metric_name in metric_names
            }
            logger.info(f"Resetting weights into {metric_weights}")
            weights_array = np.array([metric_weights[metric_name] for metric_name in metric_names])
        
        final_metric_scores = metric_array @ weights_array

        return self.evaluate(final_metric_scores, target_scores)

    def run_bayesian_optimization(self, metric_scores, target_scores):
        """Run Bayesian optimization with given parameters."""
        metric_array = np.array([metric_scores[metric_name] for metric_name in metric_scores]).T
        def wrapped_black_box_function(**metric_weights):
            return self.black_box_function(metric_array, target_scores, metric_weights)
        
        pbounds = {metric_name: (0, 1) for metric_name in metric_scores.keys()}
        optimizer = BayesianOptimization(f=wrapped_black_box_function, pbounds=pbounds, random_state=self.seed)
        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)
        
        return optimizer
    
    def load_optimizer(self, weight_params_file):
        with open(weight_params_file, "r") as f:
            self.optimizer = json.load(f)
        self.need_calibrate = False

    def calibrate(self, metrics_df, target_scores):
        """Optimize metrics weights based on given seed."""
        metric_scores = {metric_name: metrics_df[metric_name].to_numpy() for metric_name in metrics_df.columns}
        
        optimizer = self.run_bayesian_optimization(metric_scores, target_scores)
        self.optimizer = optimizer.max['params']
        self.need_calibrate = False
        logger.info("Optimizer has been calibrated!")
        
        # Evaluate if needed (for debugging)
        obj_result = self.evaluate(self.predict(metrics_df), target_scores)
        logger.debug(f"Correlation gained from the current optimizer is {obj_result} where metrics used are {metrics_df.columns}")
    
    def predict(self, metrics_df):
        if self.need_calibrate:
            logger.warning("Current optimizer has not been calibrated yet prediction is being performed. Please calibrate the optimizer first before doing prediction!")

        weights_array = np.array([self.optimizer.get(col, 0) for col in metrics_df.columns])
        final_scores = metrics_df.to_numpy() @ weights_array
        return final_scores
        
    def evaluate(self, pred, expected):
        return self.objective_fn(pred, expected)
