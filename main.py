from meta_metrics.meta_metrics import MetaMetrics

if __name__ == "__main__":
    print("initialize MetaMetrics")
    metrics_configs = [("bertscore", {"model_name": "xlm-roberta-base", "model_metric": "f1"}),
                       ("bertscore", {"model_name": "xlm-roberta-large", "model_metric": "f1"})]

    metric = MetaMetrics(metrics_configs, weights=[0.7, 0.3])
    print("score:", metric.score(["sentence1", "sentence2"], [["sentence1", "sentence2"],["sentence1", "sentence2"]]))