from typing import List, Union
from abc import ABC, abstractmethod
import torch
import numpy as np
import gc

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

EPSILON = 1e-5

class BaseMetric(ABC):
    @abstractmethod
    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        raise NotImplementedError()
    
    @classmethod
    def normalize(self, scores: List[float], min_val: float, max_val: float, invert: bool, clip: bool) -> np.ndarray:
        # Normalize to [0, 1] where greater value indicate better performance in respective metric
        if clip:
            scores = np.clip(scores, min_val, max_val)
        elif (min_val - EPSILON <= scores).any() and (scores <= max_val + EPSILON).any():
            scores = np.clip(scores, min_val, max_val)
        
        scores = (scores - min_val) / (max_val - min_val)
        if invert:
            scores = 1 - scores
            
        return scores
    
    @classmethod
    def _cleanup(self):
        """Method to handle cleanup after scoring"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Free up GPU memory if used
        gc.collect()  # Collect any lingering garbage to free memory

    @classmethod
    def run_scoring(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        """Main method to score predictions and handle initialization and cleanup."""
        result = self.score(predictions, references, sources)
        self._cleanup()
        return result
    
    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()
    
class VisionToTextBaseMetric(ABC):
    @abstractmethod
    def score(self, image_sources: List[torch.Tensor], text_predictions: List[str], text_references: Union[None, List[List[str]]]=None, text_sources: Union[None, List[str]]=None) -> List[float]:
        raise NotImplementedError()
    
    @classmethod
    def normalize(self, scores: List[float], min_val: float, max_val: float, invert: bool, clip: bool) -> np.ndarray:
        # Normalize to [0, 1] where greater value indicate better performance in respective metric
        if clip:
            scores = np.clip(scores, min_val, max_val)
        elif (min_val - EPSILON <= scores).any() and (scores <= max_val + EPSILON).any():
            scores = np.clip(scores, min_val, max_val)
        
        scores = (scores - min_val) / (max_val - min_val)
        if invert:
            scores = 1 - scores
            
        return scores
    
    @classmethod
    def _cleanup(self):
        """Method to handle cleanup after scoring"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Free up GPU memory if used
        gc.collect()  # Collect any lingering garbage to free memory

    @classmethod
    def run_scoring(self, image_sources: List[torch.Tensor], text_predictions: List[str], text_references: Union[None, List[List[str]]]=None, text_sources: Union[None, List[str]]=None) -> List[float]:
        """Main method to score predictions and handle initialization and cleanup."""
        result = self.score(image_sources, text_predictions, text_references, text_sources)
        self._cleanup()
        return result
    
    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError()