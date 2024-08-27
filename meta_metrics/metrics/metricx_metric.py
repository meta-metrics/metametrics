from meta_metrics.metrics.base_metric import BaseMetric
from meta_metrics.metrics.utils.metricx import *
import json
from torch.utils.data import Dataset
from typing import Dict, List, Union
import datasets
import os
import torch
import transformers
import uuid

class MetricXMetric(BaseMetric):
    def __init__(self, is_qe: bool, tokenizer_name: str, model_name: str, batch_size: int,
                 max_input_length: int, bf16: bool, **kwargs):
        self.reference_free = is_qe
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        self.bf16 = bf16
        if self.bf16:
            self.model = MT5ForRegression.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
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
            output_dir=".",
            bf16=self.bf16,
        )
        if self.per_device_batch_size == 1:
            self.trainer = transformers.Trainer(
                model=self.model,
                args=self.training_args,
            )
        else:
            data_collator = transformers.DataCollatorWithPadding(tokenizer=self.tokenizer, padding=True)
            self.trainer = transformers.Trainer(
                model=self.model,
                args=self.training_args,
                data_collator = data_collator
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
            if self.per_device_batch_size == 1:
                return self.tokenizer(
                    example["input"],
                    max_length=self.max_input_length,
                    truncation=True,
                    padding=False,
                )
            else:
                return self.tokenizer(
                    example["input"],
                    max_length=self.max_input_length,
                    truncation=True,
                    # padding=False,
                    padding='max_length',
                )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

        data_obj = []
        for i in range(len(hypothesis)):
            new_obj = {}
            if sources is not None:
                new_obj["source"] = sources[i]
            if hypothesis is not None:
                new_obj["hypothesis"] = hypothesis[i]
            if references is not None:
                new_obj["reference"] = references[i]
            data_obj.append(new_obj)

        input_file = str(uuid.uuid1()) + ".json"
        print(f"input file: {input_file}")
        with open(input_file, "w+") as f:
            f.write(json.dumps(data_obj) + "\n")

        ds = datasets.load_dataset("json", data_files={"test": input_file})
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=self.device,
            output_all_columns=True,
        )
        os.system(f"rm {input_file}")
        return ds

    def score(self, predictions:List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[List[str]]]=None) -> List[float]:
        ds = self.get_dataset(
            sources,
            predictions,
            references,
        )
 
        predictions, _, _ = self.trainer.predict(test_dataset=ds["test"])
        return predictions
