from meta_metrics.metrics.base_metric import BaseMetric
from huggingface_hub import login
from comet import download_model, load_from_checkpoint

class CometKiwi(BaseMetric):
    """
        args:
            metric_args (Dict): a dictionary of metric arguments

        You must obtain access to https://huggingface.co/Unbabel/wmt22-cometkiwi-da at huggingface.
        You must then go to your huggingface profile and create an access token with the following permission:
            - Read access to contents of all public gated repos you can access
        Pass this during the instantiation of the class.
    """
    def __init__(self, hf_pat):
        self.login = False
        if hf_pat:
            try:
                login(token=hf_pat)
                self.login=True

                model_path = download_model("Unbabel/wmt22-cometkiwi-da")
                self.model = load_from_checkpoint(model_path)
            except:
                print(f"Huggingface personal access token not accepted. Ensure you have proper permission and have access to wmt22-cometkiwi-da")

    def score(self, data, batch_size=8, gpus=1):
        """
            args:
                data (list of dict): Each dictionary contains 2 keys: "src" and "mt"
        """
        if self.login:
            return self.model.predict(data, batch_size=batch_size, gpus=gpus)
        return None