import csv
from datasets import load_dataset
from meta_metrics import MetaMetrics

def run(dataset, metrics):
    metric_scores = {}
    for metric_id in range(len(metrics)):
        srcs, refs, hyps = [], [], []
        metric = metrics[metric_id]["model"]
        metric_name = metrics[metric_id]["name"]
        
        scores = []
        for dataset in datasets:
            for obj in dataset:
                src, ref, hyp = obj["src"], obj["ref"], obj["mt"]
                srcs.append(src)
                refs.append(ref)
                hyps.append(hyp)

        scores.append(metric.score(hyps, refs, srcs))
        metric_scores[metric] = scores
    return metric_scores
    

if __name__ == "__main__":
    dataset_names =  ["RicardoRei/wmt-sqm-human-evaluation", "RicardoRei/wmt-mqm-human-evaluation", "RicardoRei/wmt-da-human-evaluation"]
    metrics, datasets = [], []

    metrics_configs = [
        ("bertscore", {"model_name": "xlm-roberta-base", "model_metric": "f1"}),
        # ("yisi", {"model_name": "xlm-roberta-base", "alpha": 0.8}),
        # ("comet", {"hf_token": "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"}),
        # ("xcomet", {"hf_token": "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"}),
        # ("cometkiwi", {"hf_token": "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"}),
    ]

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name, split="train")
        datasets.append(dataset)

    for metric_id in range(len(metrics_configs)):
        metric = MetaMetrics([metrics_configs[metric_id]], weights=[1])
        metrics.append({"name": metrics_configs[metric_id][0], "model":metric, "src_lang":None, "tgt_lang":None})

    scores = run(datasets, metrics)
    
    for dataset_id in range(len(datasets)):
        dataset_name = dataset_names[dataset_id].replace("/", "_")
        for metric_id in range(len(scores)):
            metric_name = metrics_configs[metric_id][0]
            with open(f"{metric_name}_{dataset_name}.tsv", "w") as tsvfile:
                tsvwriter = csv.writer(tsvfile, delimiter='\t')
                dataset = datasets[dataset_id]
                for obj in dataset:
                    obj["metric_score"] = scores[metric_name]
                    tsvwriter.writerows(obj)
