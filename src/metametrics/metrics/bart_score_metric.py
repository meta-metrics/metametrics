from evaluate import load
from typing import List, Union
import numpy as np

from metametrics.metrics.BARTScore.bart_score import BARTScorer
from metametrics.metrics.base_metric import BaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

class BARTScoreMetric(BaseMetric):
    def __init__(self, model_checkpoint: str='facebook/bart-large-cnn', max_length: int=1024,
                 agg_method: str="max", batch_size: int=8, device: str='cuda:0', **kwargs):
        self.device = device
        self.batch_size = validate_int(batch_size, valid_min=1)
        
        self.agg_method = validate_argument_list(agg_method, ["mean", "max"])

        # Lazy initialization
        self.max_length = validate_int(max_length, valid_min=1)
        self.model_checkpoint = model_checkpoint
            
    def _initialize_metric(self):
        # Actual initialization
        self.bart_scorer = BARTScorer(device=self.device, max_length=self.max_length,
                                      checkpoint=self.model_checkpoint)

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[List[str]]]=None) -> List[float]:
        self._initialize_metric()
        all_scores = self.bart_scorer.multi_ref_score(predictions, references, agg=self.agg_method, batch_size=self.batch_size)
        return all_scores
    
    def normalize(cls, scores: List[float]) -> np.ndarray:
        return super().normalize(scores, min_val=0.0, max_val=1.0, invert=False, clip=False)
