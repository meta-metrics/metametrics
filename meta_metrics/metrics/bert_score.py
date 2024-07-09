from meta_metrics.metrics import BaseMetric
from typing import List, Tuple
from evaluate import load

class BERTScore(BaseMetric):
    """
    """
    def __init__(self, args):
        self.hf_metric = load("bertscore")
        self.model_name = args["model_name"]
        self.model_metric = args["model_metric"]
        
        if self.model_metric not in ["precision", "recall", "f1"]:
            self.model_metric = "f1"

    def score(self, predictions:List[str], references:List[str]) -> List[float]:
        return self.hf_metric.compute(predictions=predictions, references=references, model_type=self.model_name)[self.model_metric]