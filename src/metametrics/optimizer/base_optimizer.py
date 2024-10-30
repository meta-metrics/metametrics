from typing import List, Union
from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    @abstractmethod
    def __init__(self, config_file):
        raise NotImplementedError()
    
    @abstractmethod
    def calibrate(self, metrics_df, target_scores):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, metrics_df):
        raise NotImplementedError()
    