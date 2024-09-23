from xgb_utils import *

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
folder_name = "output_xgb_combined"

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
drop_columns = ['id', 'model_id', 'expert_annotations_agg', 'turker_annotations_agg',
                "turker_coherence", "turker_consistency", "turker_fluency", "turker_relevance"]

xgb_json_path = "/u/anugraha/summeval/regressor_configs/xgb_config_summeval.json"
xgb_json_path2 = "/u/anugraha/summeval/regressor_configs/xgb_config_summeval_sys.json"
xgb_models = get_weight_optimization(combined_df, metrics, drop_columns, SPLITS_TEST, HUMAN_EVALS_LIST, group_system, xgb_json_path, xgb_json_path2)

for i, split in enumerate(SPLITS_TEST):
    df = combined_df.copy(True)
    for j in range(len(HUMAN_EVALS_LIST)):
        xgb_model, xgb_model_new = xgb_models[i * len(HUMAN_EVALS_LIST) + j]
        df[FINAL_HE_LIST[j]] = xgb_model.predict(df[metrics])

    train_df, test_df = train_test_split(df, test_size=split, random_state=1, stratify=df[group_system])
    
    for dat in ["cnndm", "xsum"]:
        list_sys_scores = []
        for j in range(len(HUMAN_EVALS_LIST)):
            xgb_model, xgb_model_new = xgb_models[i * len(HUMAN_EVALS_LIST) + j]
            df_copy = df.copy(True)[(df["bllm_dataset"] == dat) & (df["original_data"] == "bllm")]
            aggr_pred = df_copy.groupby(group_system).agg({FINAL_HE_LIST[j]: 'mean'}).reset_index()[FINAL_HE_LIST[j]].to_numpy().astype(np.float32).reshape(-1, 1)
            list_sys_scores.append(xgb_model_new.predict(aggr_pred))
        
        get_correlation_ours_sys(df[(df["bllm_dataset"] == dat) & (df["original_data"] == "bllm")], HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}_{dat}_overall")
        get_correlation_ours_sys(test_df[(test_df["bllm_dataset"] == dat) & (test_df["original_data"] == "bllm")], HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}_{dat}")
        get_correlation_ours_seg(df[(df["bllm_dataset"] == dat) & (df["original_data"] == "bllm")], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}_{dat}_overall")
        get_correlation_ours_seg(test_df[(test_df["bllm_dataset"] == dat) & (test_df["original_data"] == "bllm")], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}_{dat}")

    for dat in ["bllm", "summeval"]:
        list_sys_scores = []
        for j in range(len(HUMAN_EVALS_LIST)):
            xgb_model, xgb_model_new = xgb_models[i * len(HUMAN_EVALS_LIST) + j]
            df_copy = df.copy(True)[df["original_data"] == dat]
            aggr_pred = df_copy.groupby(group_system).agg({FINAL_HE_LIST[j]: 'mean'}).reset_index()[FINAL_HE_LIST[j]].to_numpy().astype(np.float32).reshape(-1, 1)
            list_sys_scores.append(xgb_model_new.predict(aggr_pred))
        get_correlation_ours_sys(df[df["original_data"] == dat], HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}_{dat}_overall")
        get_correlation_ours_sys(test_df[test_df["original_data"] == dat], HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}_{dat}")
        get_correlation_ours_seg(df[df["original_data"] == dat], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}_{dat}_overall")
        get_correlation_ours_seg(test_df[test_df["original_data"] == dat], HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}_{dat}")
        

    # Overall
    list_sys_scores = []
    for j in range(len(HUMAN_EVALS_LIST)):
        xgb_model, xgb_model_new = xgb_models[i * len(HUMAN_EVALS_LIST) + j]
        df_copy = df.copy(True)
        aggr_pred = df_copy.groupby(group_system).agg({FINAL_HE_LIST[j]: 'mean'}).reset_index()[FINAL_HE_LIST[j]].to_numpy().astype(np.float32).reshape(-1, 1)
        list_sys_scores.append(xgb_model_new.predict(aggr_pred))
    get_correlation_ours_sys(df, HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}_overall")
    get_correlation_ours_sys(test_df, HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}")
    get_correlation_ours_seg(df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}_overall")
    get_correlation_ours_seg(test_df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}")

    # get_correlation_test_other_metrics(df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}_overall")
    # get_correlation_test_other_metrics(test_df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}")
