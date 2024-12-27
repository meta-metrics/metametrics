from typing import List, Union, Optional
import sacrebleu
import numpy as np

from metametrics.metrics.base_metric import TextBaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class BLEUMetric(TextBaseMetric):
    def __init__(self, smooth_method="exp", smooth_value=None,
                 use_effective_order=True, tokenize='13a',
                 lowercase=True, **kwargs):
        self.smooth_method = validate_argument_list(smooth_method, [None, 'floor', 'add-k', 'exp'])
        self.smooth_value = smooth_value
        self.use_effective_order = validate_bool(use_effective_order)
        self.tokenize = validate_argument_list(tokenize, [None, 'zh', '13a', 'intl', 'char', 'spm', 'flores101', 'flores200'])
        self.lowercase = validate_bool(lowercase)

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        segment_scores = []
        for pred, ref in zip(predictions, references):
            score = sacrebleu.sentence_bleu(pred, ref, lowercase=self.lowercase, tokenize=self.tokenize,
                                            smooth_method=self.smooth_method, smooth_value=self.smooth_value,
                                            use_effective_order=self.use_effective_order).score
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
        if isinstance(other, BLEUMetric):
            return vars(self) == vars(other)
 
        return False