from typing import List, Dict, Any
from datasets import DatasetDict

import os
import numpy as np

from metametrics.tasks.base import MetaMetrics, MetricManager
from metametrics.optimizer import *
from metametrics.metrics import *

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class MetaMetricsText(MetaMetrics):
    def __init__(self):
        self.metric_list = []
        self.optimizer = None
        self.metric_manager = MetricManager()
    
    def add_metric(self, metric_name: str, metric_args: Dict[str, Any]):
        self.metric_manager.add_metric(metric_name, metric_args)
        
    def set_optimizer(self, optimizer_name: str, optimizer_args: Dict[str, Any]):
        self.optimizer = super().get_optimizer(optimizer_name, optimizer_args)
        
    def evaluate_metrics(self, dataset, normalize_metrics):
        all_metric_scores = []
        predictions = dataset['hyp']
        references = dataset['ref']
        sources = dataset['src']
        
        for metric in self.metrics_manager:               
            # Get the metric scores for each metric and convert it to a list if necessary
            metric_scores = list(np.array(metric.run_scoring(predictions, references, sources)))
            all_metric_scores.append(metric_scores)  # Append the metric scores to the list
            
        if normalize_metrics:
            all_metric_scores = self.metric_manager.normalize_all_scores(all_metric_scores)
        
        return all_metric_scores
        
    def calibrate(self, metrics_df, target_scores):
        self.optimizer.optimize(metrics_df, target_scores)
        self.need_calibrate = False
    
    def evaluate_task(self, metrics_df, target_scores):
        if self.need_calibrate:
            logger.error("Modification to MetaMetrics was made, calibration is needed before evaluation!")
            raise RuntimeError()
        else:
            Y_pred = self.optimizer.predict(metrics_df)
            result = self.optimizer.evaluate(Y_pred, target_scores)
            return Y_pred, result    

def run_metametrics_text(optimizer: Dict[str, Any], dataset_dict: DatasetDict, metrics_list: List[Dict[str, Any]],
                         normalize_metrics: bool, output_dir: str, overwrite_output_dir: bool):
    text_pipeline = MetaMetricsText()
    
    # Add metrics
    for metric in metrics_list:
        text_pipeline.add_metric(metric.get("metric_name"), metric.get("metric_args"))
    
    # Set optimizer
    text_pipeline.set_optimizer(optimizer.get("optimizer_name"), optimizer.get("optimizer_args"))
    
    # Eval
    train_metric_scores = text_pipeline.evaluate_metrics(dataset_dict["train"], normalize_metrics)
    
    # Calibrate
    
    # Evaluate task
