from bayes_opt import BayesianOptimization
from scipy import stats
from typing import List, Tuple
import numpy as np
import json
import os

from meta_metrics.metrics import BERTScoreMetric
from meta_metrics.metrics import BLEURT20Metric
from meta_metrics.metrics import COMETMetric
from meta_metrics.metrics import MetricXMetric
from meta_metrics.metrics import YiSiMetric

class MetaMetrics:
    """
        Args:
            metrics_configs (List[Tuple[str, dict]]): a list of tuple of metric with the metric name and arguments.
            weights (List[float]): a list of float weight assigned to each metric
    """
    def __init__(self, metrics_configs:List[Tuple[str, dict]], weights:List[float] = None):
        self.metrics_configs = metrics_configs
        self.metrics = []
        self.weights = weights

        self.init_metrics()
        assert len(self.metrics) == len(self.weights)

    def init_metrics(self):
        for i in range(len(self.metrics_configs)):
            metric_config = self.metrics_configs[i]
            metric_name = metric_config[0]
            metric_args = metric_config[1]

            if metric_name == "bertscore":
                metric = BERTScoreMetric(**metric_args)
            elif metric_name == "bleurt":
                metric = BLEURT20Metric()
            elif metric_name == "comet":
                metric = COMETMetric(comet_model="Unbabel/wmt22-comet-da", **metric_args)
            elif metric_name == "xcomet":
                metric = COMETMetric(comet_model="Unbabel/XCOMET-XXL", **metric_args)
            elif metric_name == "cometkiwi":
                metric = COMETMetric(comet_model="Unbabel/wmt22-cometkiwi-da", **metric_args)
            elif metric_name == "metricx":
                metric = MetricXMetric(**metric_args)
            elif metric_name == "yisi":
                metric = YiSiMetric(**metric_args)
            self.metrics.append(metric)

    def score(self, predictions:List[str], references:List[str], sources: List[str] = None) -> List[float]:
        overall_metric_score = None
        for i in range(len(self.metrics)):
            metric_score = np.array(self.metrics[i].score(predictions, references, sources))
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
                                print(str((key_name, metric_name)))
                    metric_res = 0
                    for k in range(len(self.metrics_configs)):
                        metric_name = self.metrics_configs[k][0]
                        metric_res += kwargs[metric_name] * np.array(cache[str((key_name, metric_name))])
                    if corr_metric == "kendall":
                        kendall = stats.kendalltau(metric_res, score)
                        # print(kendall.statistic)
                        metric_score += kendall.statistic
                    else:
                        pass
                print(metric_score.mean())
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
            print("weights:", self.weights)
            return self.weights