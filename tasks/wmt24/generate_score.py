import csv
from datasets import load_dataset
from meta_metrics import MetaMetrics
from tqdm import tqdm

def run(datasets, metrics):
    metric_scores = {}
    for metric_id in tqdm(range(len(metrics))):
        srcs, refs, hyps = [], [], []
        metric = metrics[metric_id]["model"]
        metric_name = metrics[metric_id]["name"]
        
        for dataset in datasets:
            for i in range(len(dataset)):
                obj = dataset[i]
                src, ref, hyp = obj["src"], obj["ref"], obj["mt"]
                srcs.append(src)
                refs.append(ref)
                hyps.append(hyp)

        metric_scores[metric_name] = metric.score(hyps, refs, srcs)
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

                header = ["lp","src","mt","ref","metric_score","score","system","annotators","domain","year"]
                tsvwriter.writerows(header)

                for i in range(len(dataset)):
                    obj = dataset[i]
                    obj["metric_score"] = scores[metric_name]

                    arr = []
                    for h in header:
                        arr.append(obj[h])
                    
                    tsvwriter.writerows(arr)
