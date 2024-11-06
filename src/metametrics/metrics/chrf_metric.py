from evaluate import load
from typing import List, Union, Optional
import numpy as np

from metametrics.metrics.base_metric import BaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class chrFMetric(BaseMetric):
    def __init__(self, word_order=2, eps_smoothing=True, **kwargs):
        self.word_order = validate_int(word_order, valid_min=1) # word_order == 2 means chrF++
        self.eps_smoothing = validate_bool(eps_smoothing)  # eps_smoothing means chrF++

    def score(self, predictions: List[str], references: Union[None,List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        self.hf_metric = load("chrf")
        segment_scores = []
        for pred, ref in zip(predictions, references):
            score = self.hf_metric.compute(predictions=[pred], references=[ref], word_order=self.word_order, eps_smoothing=self.eps_smoothing)['score']
            segment_scores.append(score)
        return segment_scores
    
    @property
    def min_val(self) -> Optional[float]:
        return 0.0

    @property
    def max_val(self) -> Optional[float]:
        return 100.0

    @property
    def higher_is_better(self) -> bool:
        """Indicates if a higher value is better for this metric."""
        return True

    def __eq__(self, other):
        if isinstance(other, chrFMetric):
            self_vars = {k: v for k, v in vars(self).items() if k not in ['hf_metric']}
            other_vars = {k: v for k, v in vars(other).items() if k not in ['hf_metric']}
        
            return self_vars == other_vars
 
        return False
    