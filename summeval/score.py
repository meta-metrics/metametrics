import os
import shutil
import random
import numpy as np
import pandas as pd
import torch
import argparse
import json
import logging

import evaluate

logging.disable(logging.CRITICAL)
logging.disable(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--model_checkpoint",
    #     default=None,
    #     type=str,
    #     required=True,
    #     help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    # )
    # parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    args = parser.parse_args()

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True

    print("###########################")
    # print("task:", args.task)
    print("###########################")

    # set_seed(args.seed)
    scores = {}

    all_human_metric_scores = {
        "coherence": [],
        "consistency": [],
        "fluency": [],
        "relevance": []
    }

    map_score_key = {
        "bleu": "bleu",
        "sacrebleu": "score"
    }

    metric_scores = {}
    human_metric_scores = {}
    with open("model_annotations.aligned.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            expert_annotations = obj["expert_annotations"]
            prediction = [obj["decoded"]]
            reference = [obj["references"]]

            # compute avg human metrics
            human_metrics = {}
            for human_metric_name in ["coherence", "consistency", "fluency", "relevance"]:
                for annotation in expert_annotations:
                    if human_metric_name not in human_metrics:
                        human_metrics[human_metric_name] = []
                    human_metrics[human_metric_name].append(annotation[human_metric_name])

                if human_metric_name not in human_metric_scores:
                    human_metric_scores[human_metric_name] = []
                    
                human_metric_scores[human_metric_name].append(np.mean(np.array(human_metrics[human_metric_name])))

            metrics = {}
            for metric_name in ["bleu", "sacrebleu", "meteor"]:
                metrics[metric_name] = evaluate.load(metric_name)

            for human_metric_name in ["coherence", "consistency", "fluency", "relevance"]:
                for metric_name in ["bleu", "sacrebleu", "meteor"]:
                    print(human_metric_name, metric_name)
                    metric = metrics[metric_name]
                    score = metric.compute(predictions=prediction, references=reference)[map_score_key[metric_name]]
                    print(score)
                    if metric_name not in metric_scores:
                        metric_scores[metric_name] = []
                    metric_scores[metric_name].append(score)
    
    # print(metric_scores)
    for key in metric_scores:
        print(key, len(metric_scores[key]))

    for key in human_metric_scores:
        print(key, len(human_metric_scores[key]))
