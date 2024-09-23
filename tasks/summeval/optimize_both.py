from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from utils import *

# Define human evaluations and metrics
HUMAN_EVALS_LIST = ['coherence', 'relevance']
FINAL_HE_LIST = ["final_score_coherence", "final_score_relevance"]

metrics = ['bleu', 'chrf', 'meteor', 'rouge-we1', 'rouge1', 'rouge3',
       'rouge4', 'rougel', 'bartscore_max', 'bartscore_mean', 'bleurt_max', 'bleurt_mean', 'bertscore_f1', 'bertscore_recall', 'summaqa_f1',
       'summaqa_prob']

group_system = ['original_data', 'model']

setting = "max"
drop_columns=['model', 'original_data', 'bllm_dataset']
SPLITS_TEST = [0.7]
folder_name = "output_combined"

objective = "kendall"

df1 = pd.read_csv('/u/anugraha/summeval/benchmark_llm_with_scores_aggr.csv')
df2 = pd.read_csv('/u/anugraha/summeval/summeval_with_scores.csv', sep='|')

df1["original_data"] = "bllm"
df1["bartscore_mean"] = np.array(df1["bartscore"])
df1["bartscore_max"] = np.array(df1["bartscore"])
df1["bleurt_mean"] = np.array(df1["bleurt"])
df1["bleurt_max"] = np.array(df1["bleurt"])
df1 = df1.rename(columns={"dataset": "bllm_dataset"})
df1 = df1.drop(columns=["faithfulness", "bleurt", "bartscore"])


df2["original_data"] = "summeval"
df2 = df2.drop(columns=["id", "expert_annotations_agg", "turker_annotations_agg",
                        "turker_coherence", "turker_fluency", "turker_relevance", "turker_consistency",
                        "expert_fluency", "expert_consistency"])
df2 = df2.rename(columns={'model_id': 'model', 'expert_coherence': 'coherence', 'expert_relevance': 'relevance'})
df2["bllm_dataset"] = "summeval"

combined_df = pd.concat([df1, df2], axis=0)

weights = get_weight_optimization(combined_df, metrics, drop_columns, SPLITS_TEST, HUMAN_EVALS_LIST, group_system, objective="kendall", init_points=5, n_iter=100, n_seeds=64)

for i, split in enumerate(SPLITS_TEST):
    df = combined_df.copy(True)
    for j in range(len(HUMAN_EVALS_LIST)):
        weight_params = weights[i * len(HUMAN_EVALS_LIST) + j]['params']
        logging.info(f"The weight for {HUMAN_EVALS_LIST[j]} is {weight_params}")
        df[FINAL_HE_LIST[j]] = df.apply(lambda row: sum(row[col] * weight_params.get(col, 0) for col in metrics), axis=1)

    train_df, test_df = train_test_split(df, test_size=split, random_state=1, stratify=df[group_system])
    
    for dat in ["cnndm", "xsum"]:
        get_correlation_ours(df[(df["bllm_dataset"] == dat) & (df["original_data"] == "bllm")], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"ours_{dat}_{int(split * 100)}_overall")
        get_correlation_ours(test_df[(test_df["bllm_dataset"] == dat) & (test_df["original_data"] == "bllm")], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"ours_{dat}_{int(split * 100)}")
        
        get_correlation_test_other_metrics(df[(df["bllm_dataset"] == dat) & (df["original_data"] == "bllm")], HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_{dat}_split_{int(split * 100)}_overall")
        get_correlation_test_other_metrics(test_df[(test_df["bllm_dataset"] == dat) & (test_df["original_data"] == "bllm")], HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_{dat}_split_{int(split * 100)}")

    for dat in ["bllm", "summeval"]:
        get_correlation_ours(df[df["original_data"] == dat], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"ours_{dat}_{int(split * 100)}_overall")
        get_correlation_ours(test_df[test_df["original_data"] == dat], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"ours_{dat}_{int(split * 100)}")
        
        get_correlation_test_other_metrics(df[df["original_data"] == dat], HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_{dat}_split_{int(split * 100)}_overall")
        get_correlation_test_other_metrics(test_df[test_df["original_data"] == dat], HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_{dat}_split_{int(split * 100)}")

    # Overall
    get_correlation_ours(df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"ours_{int(split * 100)}_overall")
    get_correlation_ours(test_df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"ours_{int(split * 100)}")

    get_correlation_test_other_metrics(df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}_overall")
    get_correlation_test_other_metrics(test_df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}")
