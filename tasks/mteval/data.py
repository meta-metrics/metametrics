import os
from mt_metrics_eval import data, meta_info

class MTMEDataLoader:
    def __init__(self, split_dict, data_path=None):
        """
        split_dict: A dictionary defining the splits, e.g.,
                    {
                        "train": {
                            "wmt19": ["en-de", "en-fr"],
                            "wmt20": ["en-de", "en-fr"]
                        },
                        "dev": {
                            "wmt21": ["en-de", "en-fr"]
                        },
                        "test": {
                            "wmt22": ["en-de", "en-fr"],
                            "wmt23": ["en-de", "en-fr"]
                        }
                    }
        data_path: The path where the data is stored.
        """
        self.split_dict = split_dict
        self.data_path = data_path or os.path.join(os.path.expanduser('~'), '.mt-metrics-eval', 'mt-metrics-eval-v2')
        self._validate_input()

    def _validate_input(self):
        for split, datasets in self.split_dict.items():
            for test_set, lang_pairs in datasets.items():
                if test_set not in meta_info.DATA:
                    raise ValueError(f"Test set '{test_set}' not found in meta_info.DATA.")
                for lang_pair in lang_pairs:
                    if lang_pair not in meta_info.DATA[test_set]:
                        raise ValueError(f"Language pair '{lang_pair}' not found in test set '{test_set}' in meta_info.DATA.")

    def load_data(self):
        split_data = {'train': {"output": [], "reference": [], "score": [], 'source': [], 'key': []},
                      'dev': {"output": [], "reference": [], "score": [], 'source': [], 'key': []},
                      'test': {"output": [], "reference": [], "score": [], 'source': [], 'key': []},
                     }

        for split, datasets in self.split_dict.items():
            for test_set, lang_pairs in datasets.items():
                for lang_pair in lang_pairs:
                    eval_set = data.EvalSet(name=test_set, lp=lang_pair, path=self.data_path)
                    keys, sources, outputs, references, scores = self._load_data_from_evalset(eval_set)
                    split_data[split]["output"].extend(outputs)
                    split_data[split]["reference"].extend(references)
                    split_data[split]["score"].extend(scores)
                    split_data[split]["source"].extend(sources)
                    split_data[split]["key"].extend(keys)
                    
        
        return split_data

    def _load_data_from_evalset(self, eval_set):
        g_keys, g_src, g_outputs, g_references, g_scores = [], [], [], [], []
        systems = eval_set.sys_outputs.keys()
        
        for sys_name in systems:
            keys, src, outputs, references, scores = [], [], [], [], []
            sys_outputs = eval_set.sys_outputs[sys_name]
            if sys_name in eval_set.human_sys_names:
                continue
            scorer_name = 'mqm' if 'mqm' in eval_set.human_score_names else 'wmt-raw'
            score = eval_set.Scores(level='seg', scorer=scorer_name)
            if score:
                num_segments = len(sys_outputs)
                if len(score[sys_name]) == num_segments:
                    for i in range(num_segments):
                        if score[sys_name][i] is not None:
                            outputs.append(sys_outputs[i])
                            references.append(eval_set.all_refs[eval_set.std_ref][i])
                            src.append(eval_set.src[i])
                            scores.append(score[sys_name][i])
                            keys.append(eval_set.name + '&' + eval_set.lp + '&' + sys_name)
            else:
                print("Missing MQM or WMT-RAW score")
                print(eval_set.name, eval_set.lp, sys_name)
            g_keys.append(keys)
            g_outputs.append(outputs)
            g_references.append(references)
            g_scores.append(scores)
            g_src.append(src)
        return g_keys, g_src, g_outputs, g_references, g_scores


if __name__ == "__main__":
    # Example usage:
    split_dict = {
        "train": {
            "wmt19": ["en-de", "en-ru"],
            "wmt20": ["en-de", "en-ru"],
            "wmt21.news": ["en-de", "en-ru"]
        },
        "dev": {
            "wmt22": ["en-de", "en-ru"],
            "wmt23": ["en-de", "en-ru"]
        },
    }
    data_loader = MTMEDataLoader(split_dict=split_dict)
    split_data = data_loader.load_data()
    
    # Flattened data structure
    train_output = split_data['train']["output"]
    train_reference = split_data['train']["reference"]
    train_score = split_data['train']["score"]
    
    print(len(train_output), len(train_reference), len(train_score))
    # print(len(train_output[0]), len(train_reference[0]), len(train_score[0]))
    
    dev_output = split_data['dev']["output"]
    dev_reference = split_data['dev']["reference"]
    dev_score = split_data['dev']["score"]
    
    print(len(dev_output), len(dev_reference), len(dev_score))
    # print(dev_output[0][0], dev_reference[0][0], dev_score[0][0])
    
    test_output = split_data['test']["output"]
    test_reference = split_data['test']["reference"]
    test_score = split_data['test']["score"]
    
    print(len(test_output), len(test_reference), len(test_score))