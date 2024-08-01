"""
A script used to convert WMT24 segment scores into 
WMT24 system-level scores.

The script takes in a TSV file containing segment-level
scores and outputs a TSV file containing system-level 
scores.

Example Usage:
python convert_segment_score.py \
    --input /path/to/YOURMETRIC.seg.score \
    --output /path/to/YOURMETRIC.sys.score
"""

import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the input TSV file containing segment-level scores")
    parser.add_argument("--output", type=str, required=True, help="Path to the output TSV file containing system-level scores")
    return parser.parse_args()

def main(args):
    input_df = pd.read_csv(args.input, sep="\t", dtype={"SEGMENT-SCORE": "float64"})
    print(">>>>>", input_df.columns)
    metric_name = input_df["METRIC-NAME"].unique()[0]
    input_df.drop(columns=["METRIC-NAME", "SEGMENT-NUMBER"], inplace=True)
    columns_to_groupby = "LANG-PAIR|TESTSET|DOMAIN|DOCUMENT|REFERENCE|SYSTEM-ID".split("|")
    output_df = input_df.groupby(columns_to_groupby).mean().reset_index()
    # print(output_df)
    # drop segment_id column
    output_df["METRIC-NAME"] = metric_name
    output_df.rename(columns={"SEGMENT-SCORE":"SYSTEM-SCORE"}, inplace=True)
    new_order = ["METRIC-NAME", "LANG-PAIR", "TESTSET", "DOMAIN", "REFERENCE", "SYSTEM-ID", "SYSTEM-SCORE"]
    output_df = output_df[new_order]
    
    output_df.to_csv(args.output, index=False, sep="\t")

if __name__ == "__main__":
    args = parse_args()
    main(args)