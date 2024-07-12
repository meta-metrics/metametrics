from evaluate import load
from typing import Dict, List
from .base_metric import BaseMetric

class BERTScoreMetric(BaseMetric):
    """
        args:
            metric_args (Dict): a dictionary of metric arguments
    """
    def __init__(self, metric_args:Dict):
        self.hf_metric = load("bertscore")
        self.model_name = metric_args["model_name"]
        self.model_metric = metric_args["model_metric"]
        
        if self.model_metric not in ["precision", "recall", "f1"]:
            self.model_metric = "f1"

    def score(self, predictions:List[str], references:List[List[str]]) -> List[float]:
        return self.hf_metric.compute(predictions=predictions, references=references, model_type=self.model_name)[self.model_metric]