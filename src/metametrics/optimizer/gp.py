import os
import json
from bayes_opt import BayesianOptimization
import numpy as np
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from metametrics.optimizer.base_optimizer import BaseOptimizer, OBJECTIVE_FN_MAP
from metametrics.utils.logging import get_logger
from metametrics.utils.constants import MODEL_RANDOM_SEED

logger = get_logger(__name__)

@dataclass
class GaussianProcessArguments:
    r"""
    Arguments pertaining to which GaussianProcessOptimizer will be initialized with.
    """
    objective_fn: str = field(
        metadata={"help": "Objective function name for Gaussian Process Optimization."}
    )
    init_points: int = field(
        default=5,
        metadata={"help": "Number of initial points for Bayesian optimization."}
    )
    n_iter: int = field(
        default=100,
        metadata={"help": "Number of iterations for the optimizer."}
    )
    seed: int = field(
        default=MODEL_RANDOM_SEED,
        metadata={"help": "Random seed for reproducibility."}
    )

class GaussianProcessOptimizer(BaseOptimizer):
    def __init__(self, args: GaussianProcessArguments):
        if args.objective_fn in OBJECTIVE_FN_MAP:
            self.objective_fn = OBJECTIVE_FN_MAP[args.objective_fn]
        else:
            raise ValueError(f"Objective function '{args.objective_fn}' is not recognized!")
        
        self.init_points = args.init_points
        self.n_iter = args.n_iter
        self.seed = args.seed
        self.need_calibrate = True
        self.optimizer = {}
        
    def init_from_config_file(self, config_file_path: str):
        parser = HfArgumentParser(GaussianProcessArguments)

        parsed_args = None
        if config_file_path.endswith(".yaml") or config_file_path.endswith(".yml"):
            parsed_args = parser.parse_yaml_file(os.path.abspath(config_file_path))
        elif config_file_path.endswith(".json"):
            parsed_args = parser.parse_json_file(os.path.abspath(config_file_path))
        else:
            logger.error("Got invalid dataset config path: {}".format(config_file_path))
            raise ValueError("dataset config path should be either JSON or YAML but got {} instead".format(config_file_path))
        
        self.__init__(parsed_args)

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
