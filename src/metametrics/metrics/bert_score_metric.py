from evaluate import load
from typing import List, Union
import numpy as np

from metametrics.metrics.base_metric import BaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

class BERTScoreMetric(BaseMetric):
    """
        args:
            metric_args (Dict): a dictionary of metric arguments
    """
    def __init__(self, model_name: str, model_metric: str, num_layers: int=None,
                 batch_size: int=8, nthreads: int=16, rescale_with_baseline: bool=False, **kwargs):
        self.model_name = model_name
        self.num_layers = validate_int(num_layers, valid_min=1)
        self.batch_size = validate_int(batch_size, valid_min=1)
        self.nthreads = validate_int(nthreads, valid_min=1)
        self.rescale_with_baseline = validate_bool(rescale_with_baseline)
        self.model_metric = validate_argument_list(model_metric, ["precision", "recall", "f1"])

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[List[str]]]=None) -> List[float]:
        self.hf_metric = load("bertscore")
        all_scores = self.hf_metric.compute(predictions=predictions, references=references, 
                                          model_type=self.model_name, batch_size=self.batch_size,
                                          nthreads=self.nthreads, num_layers=self.num_layers,
                                          lang="en", rescale_with_baseline=self.rescale_with_baseline)[self.model_metric]
        return all_scores
    
    def normalize(cls, scores: List[float]) -> np.ndarray:
        return super().normalize(scores, min_val=-1.0, max_val=1.0, invert=False, clip=False)
