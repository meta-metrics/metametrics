import json
import os
import pandas as pd

class EVOUNA_Dataset:
    """
    A Class to load the EVOUNA dataset

    The EVOUNA dataset consists of two subset: NQ ("Natural Questions") and TQ ("Trivia Questions")

    You can load a dataset by performing:
        tq = EVOUNA_Dataset("TQ")

    The EVOUNA Dataset contains non-compliant entry that can be filtered.
    If you want the original dataset that contains both improper and proper content, use the load_dataset() method:
        tq = EVOUNA_Dataset("TQ")
        tq_full_data = tq.load_dataset()
    
    To obtain only proper content, use process_dataset() prior to calling load_dataset():
        tq = EVOUNA_Dataset("TQ")
        tq.process_dataset()
        tq_full_data = tq.load_dataset()
    """

    # Helper Methods
    @staticmethod
    def load_json(path):
        with open(path) as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def split_by_data_improperness(lst):
        improper_list = []
        proper_list = []

        for data_entry in lst:
            if data_entry['improper']:
                improper_list.append(data_entry)
            else:
                proper_list.append(data_entry)

        return proper_list, improper_list
    
    @staticmethod
    def gold_answer_format(answer):
        if '/' in answer:
            return answer.split('/')
        
        if ' or ' in answer:
            return answer.split(' or ')
        
        return [answer]
    
    @staticmethod
    def to_int_with_exception(bool):
        if bool is None:
            return -1
        
        if bool:
            return 1
        
        return 0
    # End of Helper Methods

    def __init__(self, dataset_name):
        """
        Initialize the QA_Eval class with a specific dataset.

        Args:
            dataset_name (str): Name of the dataset ("NQ", "TQ").
        """
        assert dataset_name in ["NQ", "TQ"], 'dataset_name must either be "NQ" or "TQ"'
        if dataset_name == "NQ":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")
            self.data = EVOUNA_Dataset.load_json(os.path.join(data_dir, "NQ.json"))
        elif dataset_name == "TQ":
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(script_dir, "data")
            self.data = EVOUNA_Dataset.load_json(os.path.join(data_dir, "NQ.json"))

    def load_dataset(self):
        """
        Load the specified dataset.
        """
        
        df = pd.DataFrame(self.data)
        df = df.fillna("")
        df['golden_answer'] = df['golden_answer'].apply(EVOUNA_Dataset.gold_answer_format)

        for model in ['fid', 'gpt35', 'chatgpt', 'gpt4', 'newbing']:
            df[f'judge_{model}'] = df[f'judge_{model}'].apply(EVOUNA_Dataset.to_int_with_exception)

        return df

    def preprocess_dataset(self):
        """
        Preprocess the loaded dataset.
        """
        self.data, _ = EVOUNA_Dataset.split_by_data_improperness(self.data)