from bayes_opt import BayesianOptimization
from scipy import stats
from typing import List, Tuple
import numpy as np
import json
import os

from meta_metrics.metrics import BLEUMetric
from meta_metrics.metrics import BARTScoreMetric
from meta_metrics.metrics import BERTScoreMetric
from meta_metrics.metrics import BLEURT20Metric
from meta_metrics.metrics import chrFMetric
from meta_metrics.metrics import COMETMetric
from meta_metrics.metrics import DataStatsMetric
from meta_metrics.metrics import MetricXMetric
from meta_metrics.metrics import METEORMetric
from meta_metrics.metrics import ROUGEMetric
from meta_metrics.metrics import ROUGEWEMetric
from meta_metrics.metrics import SummaQAMetric
from meta_metrics.metrics import YiSiMetric
from meta_metrics.metrics import GEMBA_MQM

class MetaMetrics:
    """
        Args:
            metrics_configs (List[Tuple[str, dict]]): a list of tuple of metric with the metric name and arguments.
            weights (List[float]): a list of float weight assigned to each metric
            cache_mode: bool
    """
    def __init__(self, metrics_configs:List[Tuple[str, dict]], weights:List[float] = None, normalize:bool=False, cache_mode:bool=False):
        self.metrics_configs = metrics_configs
        self.metrics = []
        self.weights = weights
        self.normalize = normalize
        self.cache_mode = cache_mode

        if self.cache_mode:
            for i in range(len(self.metrics_configs)):
                metric_config = self.metrics_configs[i]
                metric_name = metric_config[0]
                metric_args = metric_config[1]
                print(f"[cache mode] initialize metric: {metric_name}")
                metric = self.get_metric(metric_name, metric_args)
                self.metrics.append(metric)
        
        if self.normalize:
            print(f"[normalize metric]")
            self.normalization_config = {
                # min, max, invert, clip
                "bertscore": (-1.0, 1.0, False, False),
                "yisi": (0.0, 1.0, False, False),
                "bleurt": (0.0, 1.0, False, True),
                "metricx": (0.0, 25.0, True, True),
                "comet": (0.0, 1.0, False, True),
                "xcomet-xl": (0.0, 1.0, False, True),
                "xcomet-xxl": (0.0, 1.0, False, True),
                "cometkiwi": (0.0, 1.0, False, True),
                "cometkiwi-xl": (0.0, 1.0, False, True),
                "cometkiwi-xxl": (0.0, 1.0, False, True),
                "gemba_mqm": (-25.0, 0.0, False, False),
                "bleu": (0.0, 1.0, False, False),
                "chrf": (0.0, 100.0, False, False),
                "meteor": (0.0, 1.0, False, False),
                "rouge": (0.0, 1.0, False, False),
                "rougewe": (0.0, 1.0, False, False),
                "summaqa": (0.0, 1.0, False, False),
                "bartscore": (0.0, 1.0, False, False),
                # "datastats": (0.0, 1.0, False, False), # TODO not sure, hence commented out
            }
            self.EPSILON = 1e-5

    def get_metric(self, metric_name, metric_args):
        print(f"get metric: {metric_name}")
        metric = None
        if metric_name == "bleu":
            metric = BLEUMetric(**metric_args)
        elif metric_name == "bartscore":
            metric = BARTScoreMetric(**metric_args)
        elif metric_name == "bertscore":
            metric = BERTScoreMetric(**metric_args)
        elif metric_name == "bleurt":
            metric = BLEURT20Metric(**metric_args)
        elif metric_name == "chrf":
            metric = chrFMetric(**metric_args)
        elif metric_name == "comet":
            metric = COMETMetric(comet_model="Unbabel/wmt22-comet-da", **metric_args)
        elif metric_name == "xcomet-xxl":
            metric = COMETMetric(comet_model="Unbabel/XCOMET-XXL", **metric_args)
        elif metric_name == "xcomet-xl":
            metric = COMETMetric(comet_model="Unbabel/XCOMET-XL", **metric_args)
        elif metric_name == "cometkiwi":
            metric = COMETMetric(comet_model="Unbabel/wmt22-cometkiwi-da", **metric_args)
        elif metric_name == "cometkiwi-xl":
            metric = COMETMetric(comet_model="Unbabel/wmt23-cometkiwi-da-xl", **metric_args)
        elif metric_name == "cometkiwi-xxl":
            metric = COMETMetric(comet_model="Unbabel/wmt23-cometkiwi-da-xxl", **metric_args)
        elif metric_name == "datastats":
            metric = DataStatsMetric(**metric_args)
        elif metric_name == "metricx":
            metric = MetricXMetric(**metric_args)
        elif metric_name == "meteor":
            metric = METEORMetric(**metric_args)
        elif metric_name == "rouge":
            metric = ROUGEMetric(**metric_args)
        elif metric_name == "rougewe":
            metric = ROUGEWEMetric(**metric_args)
        elif metric_name == "summaqa":
            metric = SummaQAMetric(**metric_args)
        elif metric_name == "yisi":
            metric = YiSiMetric(**metric_args)
        elif metric_name =="gemba_mqm":
            metric = GEMBA_MQM(**metric_args)
        return metric

    def score(self, predictions:List[str], references:List[str], sources: List[str] = None) -> List[float]:
        overall_metric_score = None
        for i in range(len(self.metrics_configs)):
            metric_config = self.metrics_configs[i]
            metric_name = metric_config[0]
            metric_args = metric_config[1]

            if self.cache_mode:
                print(f"[cache mode] get metric: {metric_name}")
                metric = self.metrics[i]
            else:
                print(f"initialize metric: {metric_name}")
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