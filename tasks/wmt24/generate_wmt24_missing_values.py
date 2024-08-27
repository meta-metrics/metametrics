# METRIC-NAME\tLANG-PAIR\tTESTSET\tDOMAIN\tDOCUMENT\tREFERENCE\tSYSTEM-ID\tSEGMENT-NUMBER\tSEGMENT-SCORE
# langs = ["en-de", "en-es", "ja-zh"]
import pandas as pd
import os
import numpy as np
from datasets import load_dataset


# TODO MODIFY THESE -- This should already contain id
ori_dataset = load_dataset("gentaiscool/wmt24-mqm", split="train")

SUBSET = ['src', 'mt'] # if qe then ['src', 'mt']

ori_df_no_filter = pd.DataFrame(ori_dataset)

# Get id when for dropped
ori_df_no_filter['id'] = range(0, len(ori_df_no_filter))
ori_df_filtered = ori_df_no_filter.dropna(subset=SUBSET)

# Get dataframe with missing values
nan_df = ori_df_no_filter[ori_df_no_filter[SUBSET].isna().any(axis=1)]

nan_df.to_csv('wmt24_missing_data.tsv', sep="\t", index=False) 