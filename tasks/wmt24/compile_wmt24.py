import argparse
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV containing containing metric scores")
    parser.add_argument("--output", type=str, required=True, help="Path to the output CSV file containing compiled scores")
    return parser.parse_args()

def main(args):
    all_scores = {}
    base_df = pd.read_csv(args.input, sep="|")
    non_metric_columns = "LANG-PAIR|TESTSET|DOMAIN|DOCUMENT|REFERENCE|SYSTEM_ID|SEGMENT_ID|lp|system|ref|src|mt|id".split("|")
    base_df_columns = base_df.columns
    for column in base_df_columns:
        if column not in non_metric_columns:
            all_scores[column] = (metric_df[column])
    
    base_df = base_df["LANG-PAIR|TESTSET|DOMAIN|DOCUMENT|REFERENCE|SYSTEM_ID|SEGMENT_ID".split("|")]
    weight_dict = {}
    weights = defaultdict(int, weight_dict)
    score = np.zeros_like(pd.DataFrame(all_scores).mean(axis=1))
    average_score = pd.DataFrame(all_scores).mean(axis=1)
    base_df["SCORE"] = average_score
    base_df.to_csv(args.output, index=False, sep="\t")

if __name__ == "__main__":
    args = parse_args()
    main(args)