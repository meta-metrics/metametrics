import os
import pandas as pd

class EQAE_Dataset:
    """
    A Class to load the "Evaluating Question Answering Evaluation" (EQAE) dataset

    The EQAE dataset consists of three subset: ["narrativeqa", "ropes", "semeval"]

    You can load a dataset by performing:
        ropes = EQAE_Dataset("ropes")
        ropes_data = ropes.load_dataset()
        
    The EQAE Datasets are in a .csv format and this class turns them into a pandas DataFrame.
    """

    # Helper Methods    

    # End of Helper Methods

    def __init__(self, dataset_name):
        """
        Initialize the QA_Eval class with a specific dataset.

        Args:
            dataset_name (str): Name of the dataset ("NQ", "TQ").
        """
        assert dataset_name in ["narrativeqa", "ropes", "semeval"], 'dataset_name must either be "narrativeqa", "ropes", or "semeval"'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        if dataset_name == "narrativeqa":
            self.data = pd.read_csv(os.path.join(data_dir, "narrativeqa_dev_predictions.csv"))
        elif dataset_name == "ropes":
            self.data = pd.read_csv(os.path.join(data_dir, "ropes_dev_predictions.csv"))
        elif dataset_name == "semeval":
            self.data = pd.read_csv(os.path.join(data_dir, "narrativeqa_dev_predictions.csv"))

    def load_dataset(self):
        return self.data