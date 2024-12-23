from typing import List, Union
from abc import ABC, abstractmethod
    
from scipy import stats

class BaseOptimizer(ABC):
    @classmethod
    @abstractmethod
    def init_from_config_dict(self, config_dict):
        raise NotImplementedError()

    @abstractmethod
    def calibrate(self, metrics_df, target_scores):
        raise NotImplementedError()
    
    @abstractmethod
    def predict(self, metrics_df):
        raise NotImplementedError()

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

def kendall_tau(y_test, y_pred):
    return stats.kendalltau(y_test, y_pred).statistic

def pearson(y_test, y_pred):
    return stats.pearsonr(y_test, y_pred).statistic

OBJECTIVE_FN_MAP = {
    "kendall": kendall_tau,
    "pearson": pearson,
    "reward_ranking_acc": reward_ranking_acc,
}
    