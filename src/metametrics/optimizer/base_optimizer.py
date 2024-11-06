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
    
from scipy import stats

def reward_ranking_acc(y_test, y_pred):
    correct_count = 0
    total_pairs = 0

    for i in range(0, len(y_pred), 2):
        # Compare the two consecutive rows
        if y_pred[i] > y_pred[i + 1] and y_test[i] > y_test[i + 1]:
            correct_count += 1
        elif y_pred[i] < y_pred[i + 1] and y_test[i] < y_test[i + 1]:
            correct_count += 1
        total_pairs += 1

    # Calculate the accuracy
    accuracy = correct_count / total_pairs
    return accuracy

OBJECTIVE_FN_MAP = {
    "kendall": stats.kendalltau,
    "pearson": stats.pearsonr,
    "reward_ranking_acc": reward_ranking_acc,
}
    