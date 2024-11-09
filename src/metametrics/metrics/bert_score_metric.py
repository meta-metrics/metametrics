from evaluate import load
from typing import List, Union, Optional
import numpy as np

from metametrics.metrics.base_metric import BaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

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
    
    @property
    def min_val(self) -> Optional[float]:
        return -1.0

    @property
    def max_val(self) -> Optional[float]:
        return 1.0

    @property
    def higher_is_better(self) -> bool:
        """Indicates if a higher value is better for this metric."""
        return True

    def __eq__(self, other):
        if isinstance(other, BERTScoreMetric):
            self_vars = {k: v for k, v in vars(self).items() if k not in ['nthreads', 'hf_metric']}
            other_vars = {k: v for k, v in vars(other).items() if k not in ['nthreads', 'hf_metric']}
        
            return self_vars == other_vars
 
        return False
