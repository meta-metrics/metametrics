from typing import List, Union
from abc import ABC, abstractmethod
import numpy as np

from metametrics.metrics import *
from metametrics.optimizer import *

class MetricManager:
    # Dictionary to store registered metric classes
    _registered_metrics = {
        "bleu": BLEUMetric,
        "bartscore": BARTScoreMetric,
        "bertscore": BERTScoreMetric,
        "bleurt": BLEURT20Metric,
        "chrf": chrFMetric,
        "comet": COMETMetric,
        "metricx": MetricXMetric,
        "meteor": METEORMetric,
        "rouge": ROUGEMetric,
        "rougewe": ROUGEWEMetric,
        "summaqa": SummaQAMetric,
        "yisi": YiSiMetric,
    }
    
    # Attempt to import GEMBA_MQM_Metric
    try:
        from metametrics.metrics.gemba_metric import GEMBA_MQM_Metric
        _registered_metrics["gemba_mqm"] = GEMBA_MQM_Metric
    except ImportError:
        pass

    # Attempt to import ClipScoreMetric
    try:
        from metametrics.metrics.clip_score_metric import ClipScoreMetric
        _registered_metrics["clipscore"] = ClipScoreMetric
    except ImportError:
        pass

    def __init__(self):
        self.list_metrics = []
    
    @classmethod
    def register_metric(self, metric_name, metric_class):
        """Registers a new metric to _registered_metrics."""
        if not issubclass(metric_class, BaseMetric) and not issubclass(metric_class, VisionToTextBaseMetric):  # Ensure it inherits from BaseMetric or a similar base class
            raise TypeError(f"Metric class `{metric_class}` must inherit from BaseMetric")
        self._registered_metrics[metric_name] = metric_class

    def add_metric(self, metric_name, metric_args):
        """Fetches the specified metric by name from _registered_metrics."""
        metric_class = self._registered_metrics.get(metric_name)
        if metric_class is None:
            raise ValueError(f"Metric name '{metric_name}' is not recognized!")
        self.list_metrics.append(metric_class(**metric_args))
        
    def __iter__(self):
        return iter(self.list_metrics)

    def normalize_all_scores(self, list_of_scores: List[List[float]]) -> np.ndarray:
        """Normalizes a list of lists of scores based on each metric's min, max, and higher_is_better values."""
        if len(list_of_scores) != len(self.list_metrics):
            raise ValueError("The number of score lists must match the number of metrics in list_metrics.")
        
        normalized_scores = []
        for metric, scores in zip(self.list_metrics, list_of_scores):
            min_val = metric.min_val if metric.min_val is not None else np.min(scores)
            max_val = metric.max_val if metric.max_val is not None else np.max(scores)
            higher_is_better = metric.higher_is_better

            # Normalize the scores based on the metric's normalization parameters
            clipped_scores = np.clip(scores, min_val, max_val)
            norm_scores = (clipped_scores - min_val) / (max_val - min_val)
            norm_scores = norm_scores if higher_is_better else 1 - norm_scores
            normalized_scores.append(norm_scores)
        
        return normalized_scores

class MetaMetrics(ABC):
    _registered_optimizers = {
        "gp": GaussianProcessOptimizer,
    }
    
    # Attempt to import XGBoostOptimizer
    try:
        from metametrics.optimizer.xgb import XGBoostOptimizer
        _registered_optimizers["xgb"] = XGBoostOptimizer
    except ImportError:
        pass

    @abstractmethod
    def add_metric(self, metric_name, **metric_args):
        """Add metric to MetaMetrics"""
        raise NotImplementedError()
    
    @classmethod
    def register_optimizer(self, optimizer_name, optimizer_class):
        """Registers a new optimizer in the _registered_optimizers."""
        if not issubclass(optimizer_class, BaseOptimizer):  # Ensure it inherits from BaseOptimizer or a similar base class
            raise TypeError(f"Optimizer class `{optimizer_class}` must inherit from BaseOptimizer")
        self._registered_optimizers[optimizer_name] = optimizer_class

    def get_optimizer(self, optimizer_name, optimizer_args):
        """Gets optimizer from the _registered_optimizers."""
        optimizer_class = self._registered_optimizers.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Optimizer name '{optimizer_name}' is not recognized!")
        return optimizer_class.init_from_config_dict(config_dict=optimizer_args)
    
    @abstractmethod
    def set_optimizer(self, optimizer_name, optimizer_args):
        """Set optimizer to MetaMetrics"""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_metrics(self, metrics_df):
        raise NotImplementedError()
    
    @abstractmethod
    def calibrate(self, metrics_df, target_scores):
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_metametrics(self, metrics_df, target_scores):
        raise NotImplementedError()
    