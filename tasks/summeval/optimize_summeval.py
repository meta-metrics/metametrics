from utils import *

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

folder_name = "output_summeval_new"
objective = "kendall"
drop_columns = ['id', 'model_id', 'expert_annotations_agg', 'turker_annotations_agg',
                "turker_coherence", "turker_consistency", "turker_fluency", "turker_relevance"]
df1 = pd.read_csv('/u/anugraha/summeval/summeval_with_scores.csv', sep='|')

weights = get_weight_optimization(df1, metrics, drop_columns, SPLITS_TEST, HUMAN_EVALS_LIST, group_system, objective="kendall", init_points=5, n_iter=100, n_seeds=64)

for i, split in enumerate(SPLITS_TEST):
    df = df1.copy(True)
    for j in range(len(HUMAN_EVALS_LIST)):
        weight_params = weights[i * len(HUMAN_EVALS_LIST) + j]['params']
        logging.info(f"The weight for {HUMAN_EVALS_LIST[j]} is {weight_params}")
        df[FINAL_HE_LIST[j]] = df.apply(lambda row: sum(row[col] * weight_params.get(col, 0) for col in metrics), axis=1)

    train_df, test_df = train_test_split(df, test_size=split, random_state=1, stratify=df[group_system])

    get_correlation_ours(df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}_overall")
    get_correlation_ours(test_df, HUMAN_EVALS_LIST, FINAL_HE_LIST, group_system, folder_name, f"split_{int(split * 100)}")

    get_correlation_test_other_metrics(df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}_overall")
    get_correlation_test_other_metrics(test_df, HUMAN_EVALS_LIST, metrics, group_system, folder_name, f"other_metrics_split_{int(split * 100)}")
