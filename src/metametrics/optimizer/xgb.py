import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from scipy import stats
import json

from xgboost import XGBRanker
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.metrics import make_scorer, accuracy_score
import os
import pandas as pd
import argparse

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

MAX_WORKERS = 16
RANDOM_STATE_DATA = 1
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class XGBoostOptimizer(BaseOptimizer):
    def __init__(self, config_file):
        try:
            with open(config_file, 'r') as f:
                self.config_content = json.load(file)
        except json.JSONDecodeError:
            raise ValueError("Currently, only JSON formatted config file is supported. Please provide a valid JSON file.")
            
        if "xgb_type" not in config_content:
            raise ValueError("`xgb_type` must be provided in the config file")
        self.xgb_type = config_content.get("xgb_type")

        # Get XGBoost model parameters
        self.fixed_model_params = config_content.get("fixed_model_params", {})
     
        # Initialize BayesSearchCV through parameter space
        self.hpm_search_mode = config_content.get("hpm_search_mode", "grid_search")
        self.fixed_param = config_content.get("fixed_param", {})
        
        self.param_space = {}
        self.bayes_n_iter = 1
        config_param_space = config_content.get("param_space", {})
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
            self.bayes_n_iter = config_content.get("bayes_n_iter")
        elif self.hpm_search_mode == "grid_search":
            # Initialize GridSearchCV through parameter space
            self.param_space = config_param_space
        else:
            raise ValueError(f"Unsupported hyperparameter search type: {self.hpm_search_mode}")
        
        self.random_state
        self.cv_split = ...
        self.select_k_features = ... # If not specified that means use all features
        self.normalize_metrics = ... # can be default or custom, custom is based on train
        
        self._initialize_xgboost_model()
        
    def _initialize_xgboost_model(self):
        if self.xgb_type == 'regressor':
            self.optimizer = XGBRegressor(**fixed_model_params)
        elif self.xgb_type == 'classifier':
            self.optimizer = XGBClassifier(**fixed_model_params)
        elif self.xgb_type == 'ranker'
            self.optimizer = XGBRanker(**fixed_model_params)
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
        
    def _perform_hpm_optimization_xgboost(self, metrics_df):
        logger.info('Running XGB Bayes Search for Seg')
        logger.info(f"Parameter space for Seg: {param_space}")
        group_kfold = GroupKFold(n_splits=10)
        hpm_search_algorithm = BayesSearchCV(estimator=self.optimizer,
                                                search_spaces=self.param_space,
                                                cv=group_kfold,
                                                n_iter=self.n_iter,
                                                random_state=self.random_state,
                                                n_jobs=-1)

        # Fit the grid search to the data
        grouping = np.repeat(np.arange(len(X_train) // 2), 2)
        hpm_search_algorithm.fit(X_train, Y_train, qid=grouping, groups=grouping)
        logger.info("Best params for seg:")
        logger.info(hpm_search_algorithm.best_params_)

        # Get the best model
        best_model = hpm_search_algorithm.best_estimator_
        logger.info("Feature importance for seg:")
        logger.info(best_model.feature_importances_)
        
        
        if self.hpm_search_mode == "bayes_search":
            logger.debug('Running XGB Bayes Search')
            logger.debug(f"Parameter space: {self.param_space}")
            self.hpm_search_algorithm = BayesSearchCV(estimator=self.optimizer,
                                                    search_spaces=self.param_space,
                                                    scoring=self.scoring,
                                                    cv=self.cv,
                                                    n_iter=self.n_iter,
                                                    random_state=seed,
                                                    n_jobs=N_JOBS)
        elif self.hpm_search_mode == "grid_search":
            logger.debug('Running XGB Grid Search')
            logger.debug(f"Parameter grid: {self.param_space}")
            self.hpm_search_algorithm = GridSearchCV(estimator=xgb_model,
                                                    param_grid=self.param_space,
                                                    scoring=self.scoring,
                                                    cv=self.cv,
                                                    n_jobs=N_JOBS)

        # Fit the grid search to the data
        self.hpm_search_algorithm.fit(X_train_new, Y_train)

        # Get the best model
        self.best_model = self.hpm_search_algorithm.best_estimator_

    def calibrate(self, metrics_df, target_scores):
        model_history = []
        performance_history = []
        metrics_df_copy = metrics_df.copy(True)

        # Initialize the XGBoost model
        for i in range(len(metrics) - 2):
            performance_history.append(ranking_acc_naive(hpm_search_algorithm.predict(X_train), Y_train))
            model_history.append(best_model)
            
            # os.makedirs(os.path.join(CUR_DIR, f"output_xgb/{exp_name}/saved_models"), exist_ok=True)
            # best_model.save_model(os.path.join(CUR_DIR, f"output_xgb/{exp_name}/saved_models/iter_{i}.json"))
            
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
        
