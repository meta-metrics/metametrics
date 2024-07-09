from meta_metrics.meta_metrics import MetaMetrics

if __name__ == "__main__":
    print("initialize MetaMetrics")
    metrics_configs = []
    weights = []
    
    metric = MetaMetrics(metrics_configs, weights)