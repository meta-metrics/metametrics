from metametrics.metametrics_tasks.metametrics_base import MetaMetrics
from src.optimizer.gp import GaussianProcessOptimizer
from metametrics.metrics import *

import os

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class MetaMetricsText(MetaMetrics):
    def __init__(self):
        self.metric_list = []
        self.optimizer = None
    
    def add_metric(self, metric_name, metric_args):
        new_metric = super().get_metric(metric_name, metric_args)
        if not isinstance(new_metric, BaseMetric):
            raise TypeError(f"MetaMetricsText can only support BaseMetric! {new_metric} is not an instance of BaseMetric!")
        if new_metric not in self.metric_list:
            logger.info(f"Adding metric {metric_name} to MetaMetricsText")
            self.metric_list.append(new_metric)
        
    def set_optimizer(self, optimizer_name, optimizer_args):
        self.optimizer = super().get_optimizer(optimizer_name, optimizer_args)
        
    def evaluate_metrics(self, dataset, **kwargs):
        overall_metric_score = None
        for metric in self.metrics_configs:               
            metric_score = np.array(metric.run_scoring(predictions, references, sources))
            if self.normalize:
                metric.normalize(metric_score)
        return overall_metric_score
        
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

