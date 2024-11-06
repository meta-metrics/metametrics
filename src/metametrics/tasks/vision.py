from metametrics.metametrics_tasks.metametrics_base import MetaMetrics
from src.optimizer.gp import GaussianProcessOptimizer

import os

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class MetaMetricsVision(MetaMetrics):
    def __init__(self):
        self.metric_list = []
        self.optimizer = None
        self.need_calibrate = True
    
    def register_metric(self, metric):
        if metric not in self.metric_list:
            self.metric_list.append(metric)
            self.need_calibrate = True
        
    def register_optimizer(self, optimizer_name, **kwargs):
        if optimizer_name == "gp":
            self.optimizer = GaussianProcessOptimizer(**kwargs)
        elif optimizer_name == "xgb":
            self.optimizer = XGBoostOptimizer(**kwargs)
    
    def load_optimizer(self, load_fn, need_calibrate):
        self.need_calibrate = need_calibrate
        
    def evaluate_metrics(self, dataset, **kwargs):
        ### PLACEHOLDER BEGIN ###
        # Placeholder to run everything; Need to clone rewardbench and install it
        def run_models(model_name, dataset, batch_size, output_dir):
            command = f"rewardbench --dataset {dataset} --model={model_name} --batch_size={batch_size} --save_all --output_dir {output_dir} --trust_remote_code"
            os.system(command)
            
        for metric in self.metric_list():
            model_name, output_dir, batch_size = metric.get('model_name'), metric.get('output_dir'), metric.get('batch_size', 1)
            run_models(model_name, dataset, batch_size, output_dir)
        ### PLACEHOLDER END ###
        
    def calibrate(self, X_train, Y_train, **kwargs):
        self.optimizer.optimize(X_train, Y_train, **kwargs)
        self.need_calibrate = False
    
    def evaluate_task(self, X_test, Y_test):
        if self.need_calibrate:
            logger.error("Modification to MetaMetrics was made, calibration is needed before evaluation!")
            raise RuntimeError()
        else:
            Y_pred = self.optimizer.predict(X_test)
            result = self.optimizer.evaluate(Y_pred, Y_test)
            return Y_pred, result
            
        
        
