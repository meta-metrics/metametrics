from .base_metric import BaseMetric
from .utils.metricx import *
from torch.utils.data import Dataset
from typing import Dict, List, Union
import torch
import transformers

class MetricXDataset(Dataset):

    def __init__(self, sources, hypothesis, references):
        self.sources = sources
        self.hypothesis = hypothesis
        self.references = references

    def __getitem__(self, index):
        if self.sources != None:
            return self.predictions[index], self.references[index], self.source[index]
        else:
            return self.predictions[index], self.references[index]

    def __len__(self):
        return len(self.sources)

class MetricXMetric(BaseMetric):
    def __init__(self, is_qe: bool, tokenizer_name: str, model_name: str, batch_size: int,
                 max_input_length: int, **kwargs):
        self.reference_free = is_qe
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = MT5ForRegression.from_pretrained(model_name)
        self.max_input_length = max_input_length

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.per_device_batch_size = batch_size // torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.per_device_batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

        self.training_args = transformers.TrainingArguments(
            per_device_eval_batch_size=self.per_device_batch_size,
            dataloader_pin_memory=False,
            # output_dir,
        )

        self.trainer = transformers.Trainer(
            model=self.model,
            args=self.training_args,
        )

    def get_dataset(self, sources:Union[List[str], None], hypothesis:List[str], references:List[str]):
        """Gets the test dataset for prediction.

        If `is_qe` is true, the input data must have "hypothesis" and "source" fields.
        If it is false, there must be "hypothesis" and "reference" fields.

        Args:
            sources: a list of sources
            hypothesis: a list of hypothesis
            references: a list of gold references
            max_input_length: The maximum input sequence length.
            device: The ID of the device to put the PyTorch tensors on.
            is_qe: Indicates whether the metric is a QE metric or not.

        Returns:
            The dataset.
        """

        def _make_input(example):
            if self.reference_free:
                example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " source: "
                    + example["source"]
                )
            else:
                example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
                )
            return example

        def _tokenize(example):
            return self.tokenizer(
                example["input"],
                max_length=self.max_input_length,
                truncation=True,
                padding=False,
            )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        ds = MetricXDataset(sources, hypothesis, references)
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.device,
            output_all_columns=True,
        )
        return ds

    def score(self, predictions:List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[List[str]]]=None) -> List[float]:
        ds = self.get_dataset(
            sources,
            predictions,
            references,
        )
 
        predictions, _, _ = self.trainer.predict(test_dataset=ds["test"])
        return predictions
