import numpy as np
import warnings
import pandas as pd

SCORES_METRIC = {
    "bertscore_precision": (-1.0, 1.0, False),
    "bertscore_f1": (-1.0, 1.0, False),
    "yisi": (0.0, 1.0, False),
    "bleurt": (0.0, 1.0, False),
    "metricx": (0.0, 25.0, True),
    "comet": (0.0, 1.0, False),
    "GEMBA_score": (-25.0, 0.0, False),
    "bleu": (0.0, 100.0, False),
    "chrf": (0.0, 100.0, False),
}

EPSILON = 1e-5
NON_SCORE_COLUMNS = ['lp', 'domain', 'year', 'id', 'human_score', 'TESTSET', 'DOMAIN', 'system']

def normalize_score(value, col_name):
    if col_name in SCORES_METRIC:
        min_val, max_val, invert = SCORES_METRIC[col_name]
        if col_name == "bleurt":
            # Special case we clip since it's neural based model
            new_value = np.clip(value, 0.0, 1.0)
        else:
            new_value = value
    elif "comet" in col_name:
        min_val, max_val, invert = SCORES_METRIC["comet"]
        # Special case we clip since it's neural based model
        new_value = np.clip(value, min_val, max_val)    
    elif "metricx" in col_name:
        min_val, max_val, invert = SCORES_METRIC["metricx"]
        # Special case we clip since it's neural based model
        new_value = np.clip(value, min_val, max_val)
        
    # Provide clip (for numerical precision)
    if min_val - EPSILON <= new_value <= max_val + EPSILON:
        new_value = np.clip(value, min_val, max_val)
    else:
        warnings.warn(f"Score {col_name} seems not valid (not between {min_val} and {max_val} before normalization)", RuntimeWarning)

    normalized_value = (new_value - min_val) / (max_val - min_val)
    if invert:
        normalized_value = 1 - normalized_value
    
    # Check if valid normalization
    if -EPSILON > normalized_value or normalized_value > 1 + EPSILON:
        warnings.warn(f"Score {col_name} seems not valid (not between 0 and 1 after normalization)", RuntimeWarning)
    
    return normalized_value
    
def validate_score(scores_df):
    for col_name in scores_df.columns:
        if col_name not in (NON_SCORE_COLUMNS + ["bleurt"]):
            is_valid = scores_df[col_name].between(0, 1).all()
            if not is_valid:
                print(f"Score {col_name} is not valid")
                
def validate_ori_score(scores_df):
    for col_name in scores_df.columns:
        if col_name not in NON_SCORE_COLUMNS:
            min_val = 0
            max_val = 0
            if col_name in SCORES_METRIC:
                min_val, max_val, _ = SCORES_METRIC[col_name]
            elif "comet" in col_name:
                min_val, max_val, _ = SCORES_METRIC["comet"]
            elif "metricx" in col_name:
                min_val, max_val, _ = SCORES_METRIC["metricx"]
            else:
                raise ValueError(f"Column name {col_name} not found!")
            
            # Identify invalid values
            invalid_values = scores_df[~scores_df[col_name].between(min_val - EPSILON, max_val + EPSILON, inclusive='both')]
            if col_name != "bleurt" and "comet" not in col_name and "metricx" not in col_name and not invalid_values.empty:
                print(f"Invalid values in column {col_name}:")
                print(invalid_values[[col_name]])

ori_df = pd.read_csv(f"output/wmt24-mqm_with_metricx_metricx_cometkiwi_cometkiwi-xl.csv")

validate_ori_score(ori_df)

normalized_df = ori_df.copy(True)

# Apply normalization to each column
for col_name in normalized_df.columns:
    if col_name not in NON_SCORE_COLUMNS:
        normalized_df[col_name] = normalized_df[col_name].apply(normalize_score, col_name=col_name)

validate_score(normalized_df)

weights = {
      # "GEMBA_score": 0.0,
      "metricx_reference_free": 0.06564649056309636,
      # "metricx-23-qe-xl-v2p0_reference_free": 0.0,
      "metricx_xxl_reference_free": 0.9904603321616574,
      "cometkiwi_reference_free": 0.1267047358620059,
      "cometkiwi-xl_reference_free": 0.05844699223607353
    }

normalized_df['SEGMENT-SCORE'] = 0
for k in weights.keys():
    print(k)
    normalized_df['SEGMENT-SCORE'] += normalized_df.apply(lambda x: x[k] * weights[k], axis=1)



normalized_df.to_csv("normalized_missing_value.csv", index=False)