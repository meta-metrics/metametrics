from meta_metrics.meta_metrics import MetaMetrics
from data import MTMEDataLoader
import json
import numpy as np

if __name__ == "__main__":
    print("initialize MetaMetrics")
    metrics_configs = [
        ("bertscore", {"model_name": "xlm-roberta-base", "model_metric": "f1"}),
        # ("yisi", {"model_name": "xlm-roberta-base", "alpha": 0.8}),
        ("comet", {"hf_token": "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"}),
        # ("xcomet", {"hf_token": "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"}),
        ("cometwiki", {"hf_token": "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"}),
    ]
    uniform_weights = [0.3334,0.3333,0.3333]
    metric = MetaMetrics(metrics_configs, weights=uniform_weights)
    print("score:", metric.score(["sentence1", "sentence2"], ["sentence1", "sentence2"], ["sentence1", "sentence2"]))

    
    split_dict = {
        "train": {
            "wmt19": ['de-cs', 'de-en', 'de-fr', 'en-cs', 'en-de', 'en-fi', 'en-gu', 'en-kk', 'en-lt', 'en-ru', 'en-zh', 'fi-en', 'fr-de', 'gu-en', 'kk-en', 'lt-en', 'ru-en', 'zh-en'],
            "wmt20": ['cs-en', 'de-en', 'en-cs', 'en-de', 'en-iu', 'en-ja', 'en-pl', 'en-ru', 'en-ta', 'en-zh', 'iu-en', 'ja-en', 'km-en', 'pl-en', 'ps-en', 'ru-en', 'ta-en', 'zh-en'],
            "wmt21.news": ['en-cs', 'en-de', 'en-ha', 'en-is', 'en-ja', 'en-ru', 'en-zh', 'cs-en', 'de-en', 'de-fr', 'fr-de', 'ha-en', 'is-en', 'ja-en', 'ru-en', 'zh-en'],
            "wmt21.tedtalks": ['en-de', 'en-ru', 'zh-en'],
            "wmt21.flores": ['bn-hi', 'hi-bn', 'xh-zu', 'zu-xh'],
        },
        "dev": {
            "wmt22": ['en-de', 'en-ru', 'zh-en', 'cs-en', 'cs-uk', 'de-en', 'de-fr', 'en-cs', 'en-hr', 'en-ja', 'en-liv', 'en-uk', 'en-zh', 'fr-de', 'ja-en', 'liv-en', 'ru-en', 'ru-sah', 'sah-ru', 'uk-cs', 'uk-en'],
            "wmt23": ['en-de', 'he-en', 'zh-en', 'cs-uk', 'de-en', 'en-cs', 'en-he', 'en-ja', 'en-ru', 'en-uk', 'en-zh', 'ja-en', 'ru-en', 'uk-en'],
        },
    }

    cache_key = json.dumps(split_dict, sort_keys=True)
    
    data_loader = MTMEDataLoader(split_dict=split_dict)
    split_data = data_loader.load_data()

    train_key = split_data['train']['key']
    train_src = split_data['train']["source"]
    train_output = split_data['train']["output"]
    train_reference = split_data['train']["reference"]
    train_score = split_data['train']["score"]
    print("Train systems: ", len(train_output))
    print("Average Length of system", np.mean([len(system) for system in train_output]))

    dev_src = split_data['dev']["source"]
    dev_output = split_data['dev']["output"]
    dev_reference = split_data['dev']["reference"]
    dev_score = split_data['dev']["score"]
    print("Dev systems:", len(dev_output))
    
    calibrated_weight = metric.calibrate('GP', train_src, train_output, train_reference, train_score, {"init_points": 100, "n_iter": 10000}, cache_key = train_key)

    print("Final Weight: ", calibrated_weight)
    
    calibrated_metric = MetaMetrics(metrics_configs, weights=calibrated_weight)
    
    for i, (hyp, ref, score) in enumerate(zip(dev_output, dev_reference, dev_score)):
        metric_result = calibrated_metric.score(hyp, ref)
        with open('result_cache.json', 'a+') as f:
            result_cache = json.load(f)
        try:
            result_cache[cache_key][i] = metric_result
        except:
            result_cache[cache_key] = {i: metric_result}
        with open('result_cache.json', 'w+') as f:
            json.dump(result_cache, f, sort_keys=True)