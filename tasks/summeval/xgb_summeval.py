from xgb_utils import *

setting = "max"

# Define human evaluations and metrics
HUMAN_EVALS_LIST = ["expert_coherence", "expert_consistency", "expert_fluency", "expert_relevance"]
FINAL_HE_LIST = ["final_score_coherence", "final_score_consistency", "final_score_fluency", "final_score_relevance"]

metrics = ['bleu', 'chrf', 'meteor', 'rouge-we1', 'rouge1', 'rouge3',
       'rouge4', 'rougel', 'bartscore_max', 'bartscore_mean', 'bleurt_max', 'bleurt_mean', 'bertscore_f1', 'bertscore_recall', 'summaqa_f1',
       'summaqa_prob']

group_system = ['model_id']

setting = "max"

SPLITS_TEST = [0.7]

folder_name = "output_xgb_summeval"
objective = "kendall"
drop_columns = ['id', 'model_id', 'expert_annotations_agg', 'turker_annotations_agg',
                "turker_coherence", "turker_consistency", "turker_fluency", "turker_relevance"]
df1 = pd.read_csv('/u/anugraha/summeval/summeval_with_scores.csv', sep='|')
xgb_json_path = "/u/anugraha/summeval/regressor_configs/xgb_config_summeval.json"
xgb_json_path2 = "/u/anugraha/summeval/regressor_configs/xgb_config_summeval_sys.json"
xgb_models = get_weight_optimization(df1, metrics, drop_columns, SPLITS_TEST, HUMAN_EVALS_LIST, group_system, xgb_json_path, xgb_json_path2)

for i, split in enumerate(SPLITS_TEST):
    df = df1.copy(True)
    list_sys_scores = []
    for j in range(len(HUMAN_EVALS_LIST)):
        xgb_model, xgb_model_new = xgb_models[i * len(HUMAN_EVALS_LIST) + j]
        df[FINAL_HE_LIST[j]] = xgb_model.predict(df[metrics])
        aggr_pred = df.copy(True).groupby(group_system).agg({FINAL_HE_LIST[j]: 'mean'}).reset_index()[FINAL_HE_LIST[j]].to_numpy().astype(np.float32).reshape(-1, 1)
        list_sys_scores.append(xgb_model_new.predict(aggr_pred))

    train_df, test_df = train_test_split(df, test_size=split, random_state=1, stratify=df[group_system])

    get_correlation_ours_sys(df, HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}_overall")
    get_correlation_ours_sys(test_df, HUMAN_EVALS_LIST, list_sys_scores, group_system, folder_name, f"split_{int(split * 100)}")
    get_correlation_ours_seg(df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}_overall")
    get_correlation_ours_seg(test_df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}")

    # get_correlation_test_other_metrics(df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}_overall")
    # get_correlation_test_other_metrics(test_df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}")
