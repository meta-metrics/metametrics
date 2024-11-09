import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from scipy import stats
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from transformers import HfArgumentParser

from xgboost import XGBRegressor, XGBClassifier, XGBRanker
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd

from metametrics.optimizer.base_optimizer import BaseOptimizer, OBJECTIVE_FN_MAP
from metametrics.utils.logging import get_logger
from metametrics.utils.constants import MODEL_RANDOM_SEED

logger = get_logger(__name__)

MAX_WORKERS = 16
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class XGBArguments:
    r"""
    Arguments pertaining to the configuration of the XGBoostOptimizer.
    """
    xgb_type: str = field(
        metadata={"help": "The type of XGBoost model to use (e.g., regressor, classifier, ranker)."}
    )
    objective_fn: str = field(
        metadata={"help": "Objective function name for XGBoost Optimization."}
    )
    scoring: Optional[str] = field(
        default=None,
        metadata={"help": "Scoring method for CV."}
    )
    fixed_model_params: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Fixed model parameters for XGBoost."}
    )
    hpm_search_mode: str = field(
        default="grid_search",
        metadata={"help": "Hyperparameter search mode (grid_search or bayes_search)."}
    )
    fixed_param: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Fixed parameters for hyperparameter tuning."}
    )
    param_space: Optional[Dict[str, Dict[str, Any]]] = field(
        default_factory=dict,
        metadata={"help": "Parameter space for BayesSearch or GridSearch."}
    )
    bayes_n_iter: int = field(
        default=10,
        metadata={"help": "Number of iterations for Bayesian search."}
    )
    random_state: Optional[int] = field(
        default=MODEL_RANDOM_SEED,
        metadata={"help": "Random seed for reproducibility."}
    )
    n_jobs: Optional[int] = field(
        default=-1,
        metadata={"help": "Number of processes used for optimization."}
    )
    cv_split: int = field(
        default=10,
        metadata={"help": "Number of cross-validation splits."}
    )
    select_k_features: Optional[int] = field(
        default=None,
        metadata={"help": "Number of features to select. If None, use all features."}
    )

class XGBoostOptimizer(BaseOptimizer):
    def __init__(self, args: XGBArguments):
        self.xgb_type = args.xgb_type

        # Get XGBoost model parameters
        self.fixed_model_params = args.fixed_model_params
     
        # Initialize BayesSearchCV through parameter space
        self.hpm_search_mode = args.hpm_search_mode
        self.fixed_param = args.fixed_param
        
        self.param_space = {}
        self.bayes_n_iter = args.bayes_n_iter
        config_param_space = args.param_space
        if self.hpm_search_mode == "bayes_search":
            for param_name, param_info in config_param_space.items():
                param_type = param_info["type"]
                low = param_info["low"]
                high = param_info["high"]
                prior = param_info["prior"] if "prior" in param_info else "uniform"
                if param_type == "Integer":
                    self.param_space[param_name] = Integer(low, high, prior=prior)
                elif param_type == "Real":
                    self.param_space[param_name] = Real(low, high, prior=prior)
                elif param_type == "Categorical":
                    step = param_info["step"] if "step" in param_info else 1
                    self.param_space[param_name] = Categorical(list(range(low, high + 1, step)))
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
        elif self.hpm_search_mode == "grid_search":
            # Initialize GridSearchCV through parameter space
            self.param_space = config_param_space
        else:
            raise ValueError(f"Unsupported hyperparameter search type: {self.hpm_search_mode}")
        
        if args.objective_fn in OBJECTIVE_FN_MAP:
            self.objective_fn = OBJECTIVE_FN_MAP[args.objective_fn]
        else:
            raise ValueError(f"Objective function '{args.objective_fn}' is not recognized!")
        self.scoring = args.scoring
        
        self.random_state = args.random_state
        self.n_jobs = args.n_jobs
        self.cv_split = args.cv_split
        self.select_k_features = args.select_k_features # If not specified that means use all features
        
        self._initialize_xgboost_model()
    
    def init_from_config_file(self, config_file_path: str):
        parser = HfArgumentParser(XGBArguments)

        parsed_args = None
        if config_file_path.endswith(".yaml") or config_file_path.endswith(".yml"):
            parsed_args = parser.parse_yaml_file(os.path.abspath(config_file_path))
        elif config_file_path.endswith(".json"):
            parsed_args = parser.parse_json_file(os.path.abspath(config_file_path))
        else:
            logger.error("Got invalid dataset config path: {}".format(config_file_path))
            raise ValueError("dataset config path should be either JSON or YAML but got {} instead".format(config_file_path))
        
        self.__init__(parsed_args)
        
    def _initialize_xgboost_model(self):
        if self.xgb_type == 'regressor':
            self.optimizer = XGBRegressor(**self.fixed_model_params)
            self.hpm_optimizer_fn = self._perform_hpm_optimization_standard
        elif self.xgb_type == 'classifier':
            self.optimizer = XGBClassifier(**self.fixed_model_params)
            self.hpm_optimizer_fn = self._perform_hpm_optimization_standard
        elif self.xgb_type == 'ranker':
            self.optimizer = XGBRanker(**self.fixed_model_params)
            self.hpm_optimizer_fn = self._perform_hpm_optimization_rank
        else:
            raise NotImplementedError(f"XGBoost of type {self.xgb_type} is currently not recognized!")
        
    def load_optimizer(self, xgb_model_path):
        # Check _estimator_type for what kind of XGBoost that is
        try:
            with open(xgb_model_path, 'r') as f:
                estimator_info = json.load(f)
                self.xgb_type = estimator_info.get('_estimator_type')
        except json.JSONDecodeError:
            raise ValueError("Only JSON formatted XGBoost models are supported. Please provide a valid JSON file.")

        if self.xgb_type is None:
            # Additional check for ranking models or raise an error if unknown
            if 'objective' in estimator_info and 'rank:' in estimator_info['objective']:
                self.xgb_type = 'ranker'
            else:
                raise ValueError("Unable to determine XGBoost model type from _estimator_type or objective.")
        
        # Initialize and load model based on the type
        self._initialize_xgboost_model()
        self.optimizer.load_model(xgb_model_path)
        self.need_calibrate = False
        
    def _perform_hpm_optimization_rank(self, metrics_df, target_scores):
        logger.info('Running XGB Bayes Search for Seg')
        logger.info(f"Parameter space for Seg: {self.param_space}")
        group_kfold = GroupKFold(n_splits=self.cv_split)
        if self.hpm_search_mode == "bayes_search":
            logger.debug('Running XGB Bayes Search')
            logger.debug(f"Parameter space: {self.param_space}")
            hpm_search_algorithm = BayesSearchCV(estimator=self.optimizer,
                                                    search_spaces=self.param_space,
                                                    scoring=self.scoring,
                                                    cv=group_kfold,
                                                    n_iter=self.bayes_n_iter,
                                                    random_state=self.random_state,
                                                    n_jobs=self.n_jobs)
        else:
            logger.debug('Running XGB Grid Search')
            logger.debug(f"Parameter space: {self.param_space}")
            hpm_search_algorithm = GridSearchCV(estimator=self.optimizer,
                                                scoring=self.scoring,
                                                param_grid=self.param_space,
                                                cv=group_kfold,
                                                n_jobs=self.n_jobs)

        # Fit the grid search to the data
        grouping = np.repeat(np.arange(len(metrics_df) // 2), 2)
        hpm_search_algorithm.fit(metrics_df, target_scores, qid=grouping, groups=grouping)

        # Get the best model
        self.optimizer = hpm_search_algorithm.best_estimator_
        
    def _perform_hpm_optimization_standard(self, metrics_df, target_scores):
        logger.info('Running XGB Bayes Search for Seg')
        logger.info(f"Parameter space for Seg: {self.param_space}")
        if self.hpm_search_mode == "bayes_search":
            logger.debug('Running XGB Bayes Search')
            logger.debug(f"Parameter space: {self.param_space}")
            self.hpm_search_algorithm = BayesSearchCV(estimator=self.optimizer,
                                                    search_spaces=self.param_space,
                                                    scoring=self.scoring,
                                                    cv=self.cv_split,
                                                    n_iter=self.bayes_n_iter,
                                                    random_state=self.random_state,
                                                    n_jobs=self.n_jobs)
        elif self.hpm_search_mode == "grid_search":
            logger.debug('Running XGB Grid Search')
            logger.debug(f"Parameter grid: {self.param_space}")
            self.hpm_search_algorithm = GridSearchCV(estimator=self.optimizer,
                                                    param_grid=self.param_space,
                                                    scoring=self.scoring,
                                                    cv=self.cv_split,
                                                    n_jobs=self.n_jobs)

        # Fit the grid search to the data
        self.hpm_search_algorithm.fit(metrics_df, target_scores)

        # Get the best model
        self.optimizer = self.hpm_search_algorithm.best_estimator_

    def calibrate(self, metrics_df, target_scores):
        model_history = []
        performance_history = []
        metrics_df_copy = metrics_df.copy(True)

        # Initialize the XGBoost model
        num_reduce_feats = 1
        if self.select_k_features is not None:
            num_reduce_feats = max(1, len(metrics_df_copy.columns) - self.select_k_features)

        for _ in range(num_reduce_feats):
            self.hpm_optimizer_fn(metrics_df_copy, target_scores)
            
            performance_history.append(self.objective_fn(self.optimizer.predict(metrics_df), target_scores))
            model_history.append(self.optimizer)

            # Get feature importances
            feature_importance = self.optimizer.feature_importances_
            importance_df = pd.DataFrame({
                'feature': metrics_df_copy.columns,
                'importance': feature_importance
            }).sort_values(by='importance', ascending=False)
            
            logger.info(f"Top Features (Importance): \n{importance_df}")
            
            # Identify least important feature to remove
            if len(metrics_df_copy.columns) <= 1:
                logger.info("Only one feature left, stopping.")
                break
            
            # Drop the least important feature
            least_important_feature = importance_df.iloc[-1]['feature']  # Get the least important feature
            metrics_df_copy = metrics_df_copy.drop(columns=[least_important_feature])
            
            logger.info(f"Dropping feature: {least_important_feature}. Number of features reduced to {len(metrics_df_copy.columns)}.\n")

        # Get the best model from the iterative process
        best_index = np.argmax(performance_history)
        self.optimizer = model_history[best_index]
        self.need_calibrate = False
    
    def predict(self, metrics_df):
        if self.need_calibrate:
            logger.warning("Current optimizer has not been calibrated yet prediction is being performed. Please calibrate the optimizer first before doing prediction!")

        final_scores = self.optimizer.predict(metrics_df)
        return final_scores
    
    def evaluate(self, pred, expected):
        return self.objective_fn(pred, expected)
        
