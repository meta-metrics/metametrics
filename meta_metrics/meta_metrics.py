from bayes_opt import BayesianOptimization
from scipy import stats
from typing import List, Tuple

from meta_metrics.metrics import BERTScoreMetric

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

    def init_metrics(self):
        for i in range(len(self.metrics_configs)):
            metric_config = self.metrics_configs[i]
            metric_name = metric_config[0]
            metric_args = metric_config[1]

            if metric_name == "bertscore":
                metric = BERTScoreMetric(metric_args)
            self.metrics.append(metric)

    def score(self, predictions:List[str], references:List[str]) -> List[float]:
        overall_metric_score = None
        for i in range(len(self.metrics)):
            metric_score = self.metrics[i].score(predictions, references)
            if i == 0:
                overall_metric_score = metric_score * self.weights[i]
            else:
                for j in range(len(overall_metric_score)):
                    overall_metric_score[j] += metric_score[j] * self.weights[i]
        return overall_metric_score

    def calibrate(self, method_name, predictions, references, human_scores, optimizer_args, corr_metric="kendall"):
        if method_name == "GP":
            def black_box_function(args):
                if corr_metric == "kendall":
                    metric_res = self.score(predictions, references)
                    kendall = stats.kendalltau(metric_res, human_scores)
                    return kendall.statistic
                else:
                    pass

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