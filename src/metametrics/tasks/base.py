from typing import List, Union
from abc import ABC, abstractmethod
from metametrics.metrics import *
from metametrics.optimizer import *

class MetaMetrics(ABC):
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
        "gemba_mqm": GEMBA_MQM_Metric,
        "clipscore": ClipScoreMetric,
    }
    
    _registered_optimizers = {
        "gp": GaussianProcessOptimizer,
        "xgb": XGBoostOptimizer,
    }
    
    @classmethod
    def register_metric(self, metric_name, metric_class):
        """Registers a new metric to _registered_metrics."""
        if not issubclass(metric_class, BaseMetric) and not issubclass(metric_class, VisionToTextBaseMetric):  # Ensure it inherits from BaseMetric or a similar base class
            raise TypeError(f"Metric class `{metric_class}` must inherit from BaseMetric")
        self._registered_metrics[metric_name] = metric_class

    @classmethod
    def get_metric(self, metric_name, metric_args):
        """Fetches the specified metric by name from _registered_metrics."""
        metric_class = self._registered_metrics.get(metric_name)
        if metric_class is None:
            raise ValueError(f"Metric name '{metric_name}' is not recognized!")
        return metric_class(**metric_args)
    
    @abstractmethod
    def add_metric(cls, metric_name, **metric_args):
        """Add metric to MetaMetrics"""
        raise NotImplementedError()
    
    @classmethod
    def register_optimizer(self, optimizer_name, optimizer_class):
        """Registers a new optimizer in the _registered_optimizers."""
        if not issubclass(optimizer_class, BaseOptimizer):  # Ensure it inherits from BaseOptimizer or a similar base class
            raise TypeError(f"Optimizer class `{optimizer_class}` must inherit from BaseOptimizer")
        self._registered_metrics[optimizer_name] = optimizer_class
        
    @classmethod
    def get_optimizer(self, optimizer_name, optimizer_args):
        """Gets optimizer from the _registered_optimizers."""
        optimizer_class = self._registered_optimizers.get(optimizer_name)
        if optimizer_class is None:
            raise ValueError(f"Optimizer name '{optimizer_name}' is not recognized!")
        return optimizer_class(**optimizer_args)
    
    @abstractmethod
    def set_optimizer(self, optimizer_name, optimizer_args):
        """Set optimizer to MetaMetrics"""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_metrics(self, dataset, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def optimize(self):
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_metametrics(self):
        raise NotImplementedError()

    