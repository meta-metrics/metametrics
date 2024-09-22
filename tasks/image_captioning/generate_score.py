import os
import pandas as pd
import numpy as np
from datasets import load_dataset

import pathlib
current_dir = pathlib.Path(__file__).parent.resolve()

from meta_metrics import MetaMetrics
    
if __name__ == "__main__":
    dataset_names =  ["Flickr8k"]
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    metrics_configs = [
        ("clipscore", {"model_name": "ViT-B/32", "is_reference_only": False, "device": "cuda"}, False),
    ]
    
    all_metric_names = "_".join(config[0] for config in metrics_configs)

    for dataset_name in dataset_names:
        if "Flickr8k" in dataset_name:
            captions = {}

            with open(cur_dir + "../data/Flickr8k/Flickr8k_text/Flickr8k.token.txt") as f_open:
                for line in f_open:
                    arr = line.split("\t")
                    image_id, image_caption = arr
                    captions[image_id] = image_caption

            images_sources = []
            caption_ids = []
            caption_map = {}
            annotations = []
            references = []
            predictions = []

            with open(cur_dir + "../data/Flickr8k/Flickr8k_text/ExpertAnnotations.txt") as f_open:
                for line in f_open:
                    arr = line.split("\t")
                    caption_map[arr[0]] = arr[1]

            with open(cur_dir + "../data/Flickr8k/Flickr8k_text/ExpertAnnotations.txt") as f_open:
                for line in f_open:
                    arr = line.split("\t")
                    image = arr[0]
                    caption_id = arr[1]
                    annotation = arr[2:]
                    avg = np.mean([int(attn) for attn in annotation])

                    images_sources.append(image)
                    caption_ids.append(caption_id)
                    predictions.append(caption_map[caption_id])
                    annotations.append(annotation)

            new_df = pd.DataFrame()
            new_df["mt"] = predictions
            new_df["src"] = images_sources

            all_metric_scores = []
            for metric_id in range(len(metrics_configs)):
                metric_name = metrics_configs[metric_id][0]                
                metric = MetaMetrics([metrics_configs[metric_id]], weights=[1])

                all_metric_scores[metric] = np.array(metric.score(images_sources, predictions, references, None))
                new_df[metric_name] = all_metric_scores[metric]
            
            new_df.to_csv(os.path.join(cur_dir, f"output_wmt24/{dataset_name}_with_{all_metric_names}.csv"), index=False)