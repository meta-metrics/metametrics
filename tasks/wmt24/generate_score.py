import os

import pandas as pd
import numpy as np
from datasets import load_dataset

from meta_metrics import MetaMetrics
    
if __name__ == "__main__":
    dataset_names =  ["wmt-sqm-human-evaluation", "wmt-mqm-human-evaluation", "wmt-da-human-evaluation"]
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    metrics_configs = [
        ##### With Reference
        # ("bertscore", {"model_name": "microsoft/deberta-xlarge-mnli", "model_metric": "precision"}, False),
        # ("bertscore", {"model_name": "microsoft/deberta-xlarge-mnli", "model_metric": "recall"}, False),
        # ("bertscore", {"model_name": "microsoft/deberta-xlarge-mnli", "model_metric": "f1"}, False),
        
        # ("yisi", {"model_name": "xlm-roberta-base", "alpha": 0.8}, False),
        # ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, False),
        # ("metricx", {"model_name": "google/metricx-23-xl-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-xl", 'max_input_length': 1024, "bf16": True}, False),
        # ("metricx", {"model_name": "google/metricx-23-large-v2p0", "batch_size": 1, 'is_qe': False, 'tokenizer_name': "google/mt5-large", 'max_input_length': 1024, "bf16": True}, False),
        # ("comet", {"hf_token": "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", "batch_size": 1}, False),
        ("xcomet-xl", {"hf_token": "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", "batch_size": 1}, False),
        # ("xcomet-xxl", {"hf_token": "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", "batch_size": 1}, False),

        ##### Reference-Free
        # ("metricx", {"model_name": "google/metricx-23-xxl-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-xxl", 'max_input_length': 1024, "bf16": True}, True),
        # ("metricx", {"model_name": "google/metricx-23-xl-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-xl", 'max_input_length': 1024, "bf16": True}, True),
        # ("metricx", {"model_name": "google/metricx-23-large-v2p0", "batch_size": 1, 'is_qe': True, 'tokenizer_name': "google/mt5-large", 'max_input_length': 1024, "bf16": True}, True),        
        # ("cometkiwi", {"hf_token": "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", "batch_size": 8}, True),
        # ("cometkiwi-xxl", {"hf_token": "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", "batch_size": 1}, True),
    ]
    
    all_metric_names = "_".join(config[0] for config in metrics_configs)

    for dataset_name in dataset_names:
        # Saves locally as CSV, if exists just read from CSV
        local_csv_path = os.path.join(cur_dir, f"output/{dataset_name}.csv")
        if not os.path.exists(local_csv_path):
            os.makedirs(os.path.join(cur_dir, "output"), exist_ok=True)
            dataset = load_dataset(f"RicardoRei/{dataset_name}", split="train")
            df = pd.DataFrame(dataset)

            # Drop rows where any of the specified columns have NaN values
            df = df.dropna(subset=['src', 'mt', 'ref'])
            df['id'] = range(0, len(df))
            df.to_csv(local_csv_path, sep='|', index=False)
        else:    
            df = pd.read_csv(local_csv_path, sep='|')
        
        predictions = df["mt"].to_list()
        references = df["ref"].to_list()
        sources = df["src"].to_list()
        print(len(predictions))
        
        new_df = pd.DataFrame()
        new_df['id'] = df['id']
                
        for metric_id in range(len(metrics_configs)):
            metric_name = metrics_configs[metric_id][0]
            if metrics_configs[metric_id][2]:
                metric_name + "_reference_free"
            metric = MetaMetrics([metrics_configs[metric_id]], weights=[1])
            new_df[metric_name] = np.array(metric.score(predictions, references, sources))

        new_df.to_csv(os.path.join(cur_dir, f"output/{dataset_name}_with_{all_metric_names}.csv"), index=False)
