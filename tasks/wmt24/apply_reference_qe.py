# METRIC-NAME\tLANG-PAIR\tTESTSET\tDOMAIN\tDOCUMENT\tREFERENCE\tSYSTEM-ID\tSEGMENT-NUMBER\tSEGMENT-SCORE
# langs = ["en-de", "en-es", "ja-zh"]
import pandas as pd
import os
import numpy as np
from datasets import load_dataset

# for lang in langs:
df = pd.read_csv("all/mqm_qe_wmt24_with_score_final_scaled.csv")
print(df.columns)


weights = {
      "GEMBA_score": 0.0,
      "metricx-23-qe-large-v2p0_reference_free": 0.06564649056309636,
      "metricx-23-qe-xl-v2p0_reference_free": 0.0,
      "metricx-23-qe-xxl-v2p0_reference_free": 0.9904603321616574,
      "wmt22-cometkiwi-da_reference_free": 0.1267047358620059,
      "wmt23-cometkiwi-da-xl_reference_free": 0.05844699223607353
    }

df["metric_name"] = "metametrics-mt-mqm-qe"
df['SEGMENT-SCORE'] = 0
for k in weights.keys():
    if k == "bleu" or k == "chrf": continue
    df['SEGMENT-SCORE'] += df.apply(lambda x: x[k] * weights[k], axis=1)




# TODO MODIFY THESE -- This should already contain id
ori_dataset = load_dataset("gentaiscool/wmt24-mqm-qe", split="train")

SUBSET = ['src', 'mt'] # if qe then ['src', 'mt']
SCORE_DF = df

ori_df_no_filter = pd.DataFrame(ori_dataset)

# Get id when for dropped
ori_df_no_filter['id'] = range(0, len(ori_df_no_filter))
ori_df_filtered = ori_df_no_filter.dropna(subset=SUBSET)

# Assign this to appropriate df
SCORE_DF["id"] = np.array(ori_df_filtered["id"].values)
print(SCORE_DF)

# Get dataframe with missing values
nan_df = ori_df_no_filter[ori_df_no_filter[SUBSET].isna().any(axis=1)]
nan_df = nan_df[["TESTSET", "DOMAIN", "DOCUMENT", "REFERENCE", "SYSTEM_ID", "SEGMENT_ID", "lp", "id", "system"]]
nan_df = nan_df.rename(columns={"SEGMENT_ID": "SEGMENT_NUMBER"})
nan_df["SEGMENT-SCORE"] = 0 # Assign all to 0

# Sanity check
print(len(nan_df), len(ori_df_filtered), len(ori_dataset), len(SCORE_DF))
print(nan_df.columns, SCORE_DF.columns)
assert(len(nan_df) + len(ori_df_filtered) == len(ori_dataset), len(SCORE_DF))

submission_df = pd.concat([SCORE_DF, nan_df], ignore_index=True)
submission_df = submission_df.sort_values(by="id")

submission_df["metric_name"] = "metametrics-mt-mqm-qe"
new_order = ['metric_name', 'lp', 'TESTSET', 'DOMAIN', "DOCUMENT","REFERENCE","SYSTEM_ID", "SEGMENT_NUMBER", "SEGMENT-SCORE"]
submission_df = submission_df[new_order]

os.system("mkdir -p submission/metametrics_mt_mqm_qe_kendall")
submission_df.rename(columns={"metric_name":"METRIC-NAME", "lp":"LANG-PAIR","SYSTEM_ID": "SYSTEM-ID", "SEGMENT_NUMBER":"SEGMENT-NUMBER"}, inplace=True)
submission_df.to_csv('submission/metametrics_mt_mqm_qe_kendall/metametrics_mt_mqm_qe_kendall.seg.score', sep="\t", index=False) 