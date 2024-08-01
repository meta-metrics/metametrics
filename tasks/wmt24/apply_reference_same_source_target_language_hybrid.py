# METRIC-NAME\tLANG-PAIR\tTESTSET\tDOMAIN\tDOCUMENT\tREFERENCE\tSYSTEM-ID\tSEGMENT-NUMBER\tSEGMENT-SCORE
# langs = ["en-de", "en-es", "ja-zh"]
import pandas as pd
import os
import numpy as np
from datasets import load_dataset

# for lang in langs:
df = pd.read_csv("all/mqm_wmt24_with_score_final_scaled.csv")
print(df.columns)

weights_en_de = {
      "bertscore_f1": 0.0,
      "bertscore_precision": 0.0,
      "bleu": 0.0,
      "bleurt": 0.0,
      "chrf": 0.0,
      "metricx-23-large-v2p0": 0.0,
      "metricx-23-xl-v2p0": 0.0,
      "metricx-23-xxl-v2p0": 1.0,
      "wmt22-comet-da": 0.2055307813370211,
      "xcomet-xl": 0.27327721603913696,
      "yisi": 0.0
    }
weights_en_es = {
      "bertscore_f1": 0.0,
      "bertscore_precision": 0.0,
      "bleu": 0.0,
      "bleurt": 0.0,
      "chrf": 0.0,
      "metricx-23-large-v2p0": 0.0,
      "metricx-23-xl-v2p0": 0.0,
      "metricx-23-xxl-v2p0": 1.0,
      "wmt22-comet-da": 0.2055307813370211,
      "xcomet-xl": 0.27327721603913696,
      "yisi": 0.0
    }
weights_ja_zh = {
      "bertscore_f1": 0.0,
      "bertscore_precision": 0.0,
      "bleu": 0.0,
      "bleurt": 0.0,
      "chrf": 0.0,
      "metricx-23-large-v2p0": 0.0,
      "metricx-23-xl-v2p0": 0.16295300667868262,
      "metricx-23-xxl-v2p0": 1.0,
      "wmt22-comet-da": 0.0,
      "xcomet-xl": 1.0,
      "yisi": 0.0
    }
weights_others = {
      "bertscore_f1": 0.0,
      "bertscore_precision": 0.0,
      "bleu": 0.0,
      "bleurt": 0.0,
      "chrf": 0.0,
      "metricx-23-large-v2p0": 0.0,
      "metricx-23-xl-v2p0": 0.0,
      "metricx-23-xxl-v2p0": 1.0,
      "wmt22-comet-da": 0.2055307813370211,
      "xcomet-xl": 0.27327721603913696,
      "yisi": 0.0
    }

df['SEGMENT-SCORE'] = 0

for i, row in df.iterrows():
    if i % 100000 == 0:
        print(">>>", i, len(df))

    for k in weights_en_de.keys():
        if k == "bleu" or k == "chrf": continue
        if row["lp"] == "en-de":
            df.loc[i, 'SEGMENT-SCORE'] += row[k] * weights_en_de[k]
        elif row["lp"] == "en-es":
            df.loc[i, 'SEGMENT-SCORE'] += row[k] * weights_en_es[k]
        elif row["lp"] == "ja-zh":
            df.loc[i, 'SEGMENT-SCORE'] += row[k] * weights_ja_zh[k]
        else:
            df.loc[i, 'SEGMENT-SCORE'] += row[k] * weights_others[k]


# TODO MODIFY THESE -- This should already contain id
ori_dataset = load_dataset("gentaiscool/wmt24-mqm", split="train")

SUBSET = ['src', 'mt', 'ref'] # if qe then ['src', 'mt']
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

missing_df = pd.read_csv("normalized_missing_value.csv")
print("missing df:", len(missing_df), "nan_df:", len(nan_df))
print(missing_df["SEGMENT-SCORE"])
nan_df["SEGMENT-SCORE"] = missing_df["SEGMENT-SCORE"].tolist()
print(">", nan_df["SEGMENT-SCORE"])

submission_df = pd.concat([SCORE_DF, nan_df], ignore_index=True)
submission_df = submission_df.sort_values(by="id")


submission_df["metric_name"] = "metametrics-mt-mqm-lang-hybrid"
new_order = ["metric_name", 'lp', 'TESTSET', 'DOMAIN', "DOCUMENT","REFERENCE","SYSTEM_ID", "SEGMENT_NUMBER", "SEGMENT-SCORE"]
submission_df = submission_df[new_order]


os.system("mkdir -p submission/metametrics_mt_mqm_same_source_target_hybrid_kendall")
submission_df.rename(columns={"metric_name":"METRIC-NAME", "lp":"LANG-PAIR","SYSTEM_ID": "SYSTEM-ID", "SEGMENT_NUMBER":"SEGMENT-NUMBER"}, inplace=True)
submission_df.to_csv('submission/metametrics_mt_mqm_same_source_target_hybrid_kendall/metametrics_mt_mqm_same_source_target_hybrid_kendall.seg.score', sep="\t", index=False)