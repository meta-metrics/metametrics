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
    input_df = pd.read_csv(args.input, sep="\t")
    columns_to_groupby = "LANG-PAIR|TESTSET|DOMAIN|DOCUMENT|REFERENCE|SYSTEM_ID".split("|")
    output_df = input_df.groupby(columns_to_groupby).mean().reset_index()
    # drop segment_id column
    output_df = output_df.drop("SEGMENT_ID", axis=1)
    output_df.to_csv(args.output, index=False, sep="\t")

if __name__ == "__main__":
    args = parse_args()
    main(args)