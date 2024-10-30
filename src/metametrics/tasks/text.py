from metametrics.metametrics_tasks.metametrics_base import MetaMetrics
from src.optimizer.gp import GaussianProcessOptimizer
from metametrics.metrics import *

import os

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

def ranking_acc(y_test, y_pred):
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

class MetaMetricsText(MetaMetrics):
    def __init__(self):
        self.metric_list = []
        self.optimizer = None
    
    def add_metric(self, metric_name, metric_args):
        new_metric = super().get_metric(metric_name, metric_args)
        if not isinstance(new_metric, BaseMetric):
            raise TypeError(f"MetaMetricsText can only support BaseMetric! {new_metric} is not an instance of BaseMetric!")
        if new_metric not in self.metric_list:
            logger.info(f"Adding metric {metric_name} to MetaMetricsText")
            self.metric_list.append(new_metric)
        
    def set_optimizer(self, optimizer_name, optimizer_args):
        self.optimizer = super().get_optimizer(optimizer_name, optimizer_args)
        
    def evaluate_metrics(self, dataset, **kwargs):
        overall_metric_score = None
        for metric in self.metrics_configs:               
            metric_score = np.array(metric.run_scoring(predictions, references, sources))
            if self.normalize:
                metric.normalize(metric_score)
        return overall_metric_score
        
    def calibrate(self, X_train, Y_train, **kwargs):
        self.optimizer.optimize(X_train, Y_train, **kwargs)
        self.need_calibrate = False
    
    def evaluate_task(self, X_test, Y_test):
        if self.need_calibrate:
            logger.error("Modification to MetaMetrics was made, calibration is needed before evaluation!")
            raise RuntimeError()
        else:
            Y_pred = self.optimizer.predict(X_test)
            result = self.optimizer.evaluate(Y_pred, Y_test)
            return Y_pred, result


    def score_vl(self, image_sources:List[torch.Tensor], text_predictions:List[str], text_references:List[str], text_sources: List[str] = None) -> List[float]:
        overall_metric_score = None
        for i in range(len(self.metrics_configs)):
            metric_config = self.metrics_configs[i]
            metric_name = metric_config[0]
            metric_args = metric_config[1]

            if self.cache_mode:
                logger.info(f"[cache mode] get metric: {metric_name}")
                metric = self.metrics[i]
            else:
                logger.info(f"initialize metric: {metric_name}")
                metric = self.get_metric(metric_name, metric_args)
                
            metric_score = np.array(metric.score(image_sources, text_predictions, text_references, text_sources))

            if self.normalize:
                _min, _max, _invert, _clip = self.normalization_config[metric_name]
                if _clip:
                    metric_score = np.clip(metric_score, _min, _max)
                
                if (_min - self.EPSILON <= metric_score).any() and (metric_score <= _max + self.EPSILON).any():
                    metric_score = np.clip(metric_score, _min, _max)
                
                metric_score = (metric_score - _min) / (_max - _min)
                if _invert:
                    metric_score = 1 - metric_score

            del metric # for efficiency

            if i == 0:
                overall_metric_score = metric_score * self.weights[i]
            else:
                overall_metric_score += metric_score * self.weights[i]
        return overall_metric_score

    def score(self, predictions:List[str], references:List[str], sources: List[str] = None) -> List[float]:
        overall_metric_score = None
        for i in range(len(self.metrics_configs)):
            metric_config = self.metrics_configs[i]
            metric_name = metric_config[0]
            metric_args = metric_config[1]

            if self.cache_mode:
                logger.info(f"[cache mode] get metric: {metric_name}")
                metric = self.metrics[i]
            else:
                logger.info(f"initialize metric: {metric_name}")
                metric = self.get_metric(metric_name, metric_args)
                
            metric_score = np.array(metric.score(predictions, references, sources))

            if self.normalize:
                _min, _max, _invert, _clip = self.normalization_config[metric_name]
                if _clip:
                    metric_score = np.clip(metric_score, _min, _max)
                
                if (_min - self.EPSILON <= metric_score).any() and (metric_score <= _max + self.EPSILON).any():
                    metric_score = np.clip(metric_score, _min, _max)
                
                metric_score = (metric_score - _min) / (_max - _min)
                if _invert:
                    metric_score = 1 - metric_score

            del metric # for efficiency

            if i == 0:
                overall_metric_score = metric_score * self.weights[i]
            else:
                overall_metric_score += metric_score * self.weights[i]
        return overall_metric_score


    def calibrate(self, method_name, sources, predictions, references, human_scores, optimizer_args, corr_metric="kendall", cache_key = None):
        cache = {}
        cache_file_path = 'meta-metrics_cache.json'
        if cache_key is not None:
            if not os.path.isfile(cache_file_path):
                with open(cache_file_path, 'w+') as f:
                    json.dump(cache, f)
            with open(cache_file_path, 'r') as f:
                cache_file = json.load(f)
                cache = cache_file
        
        if method_name == "GP":
            def black_box_function(**kwargs):
                metric_score = 0
                for i, (src, pred, ref, score) in enumerate(zip(sources, predictions, references, human_scores)):
                    key_name = cache_key[i][0]
                    for k in range(len(self.metrics_configs)):
                        metric_name = self.metrics_configs[k][0]
                        if str((key_name, metric_name)) not in cache:
                            metric_score = np.array(self.metrics[k].score(pred, ref, src))
                            cache[str((key_name, metric_name))] = metric_score.tolist()
                            if cache_key is not None:
                                cache_file = cache
                                with open(cache_file_path, 'w') as f:
                                    json.dump(cache_file, f)
                                logger.info(str((key_name, metric_name)))
                    metric_res = 0
                    for k in range(len(self.metrics_configs)):
                        metric_name = self.metrics_configs[k][0]
                        metric_res += kwargs[metric_name] * np.array(cache[str((key_name, metric_name))])
                    if corr_metric == "kendall":
                        kendall = stats.kendalltau(metric_res, score)
                        # logger.info(kendall.statistic)
                        metric_score += kendall.statistic
                    else:
                        pass
                logger.info(metric_score.mean())
                return metric_score

            # Bounded region of parameter space
            pbounds = {}
            for i in range(len(self.metrics)):
                pbounds[f"{self.metrics_configs[i][0]}"] = (0,1)

            optimizer = BayesianOptimization(
                f=black_box_function,
                pbounds=pbounds,
                random_state=1,
            )
            optimizer.maximize(
                init_points=optimizer_args["init_points"],
                n_iter=optimizer_args["n_iter"],
            )
            self.weights = []
            for i in range(len(self.metrics_configs)):
                metric_name = self.metrics_configs[i][0]
                self.weights.append(optimizer.max["params"][metric_name])
            logger.info("weights:", self.weights)
            return self.weights
            
        
        
