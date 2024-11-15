import pandas as pd
import argparse
import os
import json
import logging

logging.basicConfig(level=logging.INFO)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
INDEXING_CSV_PATH = os.path.join(CUR_DIR, "saved_results", "rb_indexing.csv")
EXAMPLE_MODEL_JSON = os.path.join(CUR_DIR, "saved_results", "example_models.json")
EXAMPLE_WEIGHT_JSON = os.path.join(CUR_DIR, "saved_results", "example_weights.json")
DEFAULT_DATASET = 'allenai/reward-bench'

SAVED_WEIGHTS = {'GRM-Gemma-2B-rewardmodel-ft_all': 0.1144232211410179,
                'GRM-Llama3-8B-rewardmodel-ft_all': 0.21877515451343996,
                'Skywork-Reward-Llama-3.1-8B_all': 1.0,
                'internlm2-1_8b-reward_all': 0.22148920242377254,
                'internlm2-7b-reward_all': 0.5482172828914557}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 447,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

EXAMPLE_TRACKER = {
    "alpacaeval-easy": 0,
    "alpacaeval-length": 0,
    "alpacaeval-hard": 0,
    "mt-bench-easy": 0,
    "mt-bench-med": 0,
    "mt-bench-hard": 0,
    "math-prm": 0,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 0,
    "refusals-offensive": 0,
    "llmbar-natural": 0,
    "llmbar-adver-neighbor": 0,
    "llmbar-adver-GPTInst": 0,
    "llmbar-adver-GPTOut": 0,
    "llmbar-adver-manual": 0,
    "xstest-should-refuse": 0,
    "xstest-should-respond": 0,
    "donotanswer": 0,
    "hep-cpp": 0,
    "hep-go": 0,
    "hep-java": 0,
    "hep-js": 0,
    "hep-python": 0,
    "hep-rust": 0,
}

def aggregate_results(model_names, json_paths):
    all_dfs = []
    for model_name, json_path in zip(model_names, json_paths):
        data = []
        jsonl_path = os.path.join(CUR_DIR, json_path)
        with open(jsonl_path, 'r') as f:
            for k, line in enumerate(f):
                record = json.loads(line)
                # Extract 'chosen' and 'rejected' values
                chosen = record.get("chosen", [None])[0]
                rejected = record.get("rejected", [None])[0]
                if 2 * k >= len(data):
                    data.append({})
                    data.append({})
                data[2 * k].update({f"{model_name}": chosen})
                data[2 * k + 1].update({f"{model_name}": rejected})
        
        current_df = pd.DataFrame(data)
        all_dfs.append(current_df)
    
    test_df = pd.concat(all_dfs, axis=1, join='inner')
    test_df['ref'] = [(i + 1) % 2 for i in range(len(test_df))]
    
    # Scale min and max
    test_df = (test_df - test_df.min()) / (test_df.max() - test_df.min())

    return test_df


def ranking_acc(y_test, y_pred):
    correct_count = 0
    total_pairs = 0
    result_subset = {}
    result_big_subset = {}
    copy_counter = EXAMPLE_TRACKER.copy()
    
    indexing = pd.read_csv(INDEXING_CSV_PATH)['subset'].to_list()
    for j in range(0, len(y_pred) // 2):
        # Compare the two consecutive rows
        i = 2 * j
        is_correct = False
        if y_pred[i] > y_pred[i + 1] and y_test[i] > y_test[i + 1]:
            is_correct = True
        elif y_pred[i] < y_pred[i + 1] and y_test[i] < y_test[i + 1]:
            is_correct = True
        
        sub_name = indexing[j]

        bigger_subset = None
        for k, v in SUBSET_MAPPING.items():
            if sub_name in v:
                bigger_subset = k
        
        assert(bigger_subset is not None)

        copy_counter[sub_name] += 1

        if sub_name not in result_subset:
            # Correct, count
            result_subset[sub_name] = [1 if is_correct else 0, 1]
        else:
            result_subset[sub_name][1] += 1
            result_subset[sub_name][0] += (1 if is_correct else 0)
            
        if bigger_subset not in result_big_subset:
            # Correct, count
            result_big_subset[bigger_subset] = [1 if is_correct else 0, 1]
        else:
            result_big_subset[bigger_subset][1] += 1
            result_big_subset[bigger_subset][0] += (1 if is_correct else 0)
            
        correct_count += (1 if is_correct else 0)
        
        total_pairs += 1

    assert(EXAMPLE_COUNTS == copy_counter)

    # Calculate the accuracy
    accuracy = correct_count / total_pairs
    submission_json = {}
    
    logging.info(f"======== RESULT ========")
    logging.info(f"Overall accuracy: {accuracy * 100:.3f}")
    for key, value in result_subset.items():
        logging.info(f"For {key}: {value[0] / value[1] * 100:.3f}")
        submission_json[key] = value[0] / value[1]
    logging.info(f"=======================")
    
    submission_json['model'] = "meta-metrics/MetaMetrics-RM-v1.0"
    submission_json['model_type'] = "Custom Classifier"
    
    with open(os.path.join(CUR_DIR, "saved_results", "submission.json"), "w") as f:
        json.dump(submission_json, f, indent=4)
    
    return accuracy, result_subset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reproduce', default=True, action=argparse.BooleanOptionalAction, help="Reproduce benchmark existing saved result")
    parser.add_argument('--weight_json_path', required=False, default=EXAMPLE_WEIGHT_JSON, type=str, help="JSON path for the weights")
    parser.add_argument('--models_json_path', required=False, default=EXAMPLE_MODEL_JSON, type=str, help="JSON path for the models used")
    args = parser.parse_args()
    
    # Run all models
    if args.reproduce:
        # Gather into pandas dataset
        test_df = pd.read_csv(os.path.join(CUR_DIR, "saved_results", "reward_bench_scores_with_prediction.tsv"), sep='\t')
        
        # Evaluate
        y_pred = test_df.apply(lambda row: sum(row[col] * SAVED_WEIGHTS.get(col, 0) for col in SAVED_WEIGHTS.keys()), axis=1)
        test_df['pred'] = y_pred
    else:
        from src.tasks.rewardbench import RewardBenchTask
        
        reward_bench_task = RewardBenchTask(need_calibrate=False)
        
        with open(args.weights_json_path, 'r') as wf, open(args.models_json_path, 'r') as mf:
            weights_dict = json.load(wf)
            models_dict = json.load(mf)
        
        for key, value in models_dict.items():
            reward_bench_task.add_metric(value)
            
        reward_bench_task.evaluate_metrics(DEFAULT_DATASET)
        
        # Gather into pandas dataset
        json_paths = [models_dict[metric_name]['json_path'] for metric_name in models_dict.keys()]
        test_df = aggregate_results(models_dict.keys(), json_paths)
        
        # Evaluate
        y_pred = test_df.apply(lambda row: sum(row[col] * weights_dict.get(col, 0) for col in weights_dict.keys()), axis=1)
        test_df['pred'] = y_pred

    ranking_acc(test_df['ref'].to_list(), y_pred.tolist())