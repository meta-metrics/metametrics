import os
from typing import Any, Dict, List, Optional, Literal, Union
from dataclasses import dataclass, field
import numpy as np
from datasets import Dataset, IterableDataset, DatasetDict, load_dataset, load_from_disk, concatenate_datasets
from transformers import HfArgumentParser

from metametrics.utils.logging import get_logger
from metametrics.utils.constants import (
    DATASET_RANDOM_SEED, FILEEXT2TYPE, CACHE_DIR, HF_TOKEN,
    TEXT_SRC, TEXT_HYP, TEXT_REF, IMG_SRC, CHOSEN, REJECTED
)

logger = get_logger(__name__)

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """
    # basic configs
    load_from: Literal["hf_hub", "file"]
    dataset_name: str
    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    num_samples: Optional[int] = None
    # text columns
    text_src: Optional[str] = TEXT_SRC
    text_hyp: Optional[str] = TEXT_HYP
    text_ref: Optional[str] = TEXT_REF
    # vision columns
    img_src: Optional[str] = IMG_SRC
    # reward columns
    chosen: Optional[str] = CHOSEN
    rejected: Optional[str] = REJECTED

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))

@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "A mapping of dataset name(s) and their arguments to use for training."},
    )
    eval_dataset: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={"help": "A mapping of dataset name(s) and their arguments to use for evaluation."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the development set, should be an integer or a float in range `[0,1)`."},
    )
    seed: int = field(
        default=DATASET_RANDOM_SEED,
        metadata={"help": "Seed for dataset."},
    )

    def __post_init__(self):
        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

def get_dataset_list(raw_dataset_attr_dict: Optional[Dict[str, Any]]) -> List[DatasetAttr]:
    r"""
    Gets the attributes of the datasets.
    """
    dataset_list = []
    for name, raw_data_attr in raw_dataset_attr_dict.items():
        # Get location of the dataset
        if "hf_hub_url" in raw_data_attr:
            dataset_attr = DatasetAttr("hf_hub", dataset_name=raw_data_attr["hf_hub_url"])
        else:
            dataset_attr = DatasetAttr("file", dataset_name=raw_data_attr["file_name"])

        # Set basic attributes of the dataset
        dataset_attr.set_attr("subset", raw_data_attr.get("subset"))
        dataset_attr.set_attr("split", raw_data_attr.get("split", "train"))
        dataset_attr.set_attr("folder", raw_data_attr.get("folder"))
        dataset_attr.set_attr("num_samples", raw_data_attr.get("num_samples"))
        
        # Set the colum names need to be retrieved
        column_names = [TEXT_SRC, TEXT_HYP, TEXT_REF, IMG_SRC, CHOSEN, REJECTED]
        for col in column_names:
            dataset_attr.set_attr(col, raw_data_attr.get(col))

        dataset_list.append(dataset_attr)

    return dataset_list

def _load_single_dataset(dataset_attr: DatasetAttr, data_args: DataArguments) -> Union[Dataset, IterableDataset]:
    r"""
    Loads a single dataset and aligns it to the standard format.
    """
    logger.info("Loading dataset {}...".format(dataset_attr))
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from == "hf_hub":
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder
    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
                if data_path is None:
                    data_path = FILEEXT2TYPE.get(file_name.split(".")[-1], None)
                elif data_path != FILEEXT2TYPE.get(file_name.split(".")[-1], None):
                    raise ValueError("File types should be identical.")
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
            data_path = FILEEXT2TYPE.get(local_path.split(".")[-1], None)
        else:
            raise ValueError("File {} not found.".format(local_path))

        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))
    else:
        raise NotImplementedError("Unknown load type: {}.".format(dataset_attr.load_from))

    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_attr.split,
        cache_dir=CACHE_DIR,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    
    # Define a dictionary to map custom column names to default names
    default_columns = {
        TEXT_SRC: getattr(dataset_attr, TEXT_SRC, None),
        TEXT_HYP: getattr(dataset_attr, TEXT_HYP, None),
        TEXT_REF: getattr(dataset_attr, TEXT_REF, None),
        IMG_SRC: getattr(dataset_attr, IMG_SRC, None),
        CHOSEN: getattr(dataset_attr, CHOSEN, None),
        REJECTED: getattr(dataset_attr, REJECTED, None),
    }

    # Rename columns in the dataset to conform to default names
    for default_name, custom_name in default_columns.items():
        if custom_name is not None and custom_name in dataset.column_names and custom_name != default_name:
            dataset = dataset.rename_column(custom_name, default_name)

    # Retain only the specified columns
    column_names_to_keep = [col for col in default_columns.keys() if default_columns[col] is not None]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in column_names_to_keep])

    target_num = dataset_attr.num_samples
    indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
    target_num -= len(indexes)
    if target_num > 0:
        expand_indexes = np.random.choice(len(dataset), target_num)
        indexes = np.concatenate((indexes, expand_indexes), axis=0)

    assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
    dataset = dataset.select(indexes)
    logger.info("Sampled {} examples from dataset {}.".format(dataset_attr.num_samples, dataset_attr))

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return dataset

def merge_dataset(raw_dataset_attr_dict: Optional[Dict[str, Any]], data_args: DataArguments) -> Optional[Union[Dataset, IterableDataset]]:
    r"""
    Gets the merged datasets in the standard format.
    """
    if raw_dataset_attr_dict is None:
        return None

    datasets = []
    for dataset_attr in get_dataset_list(raw_dataset_attr_dict):
        datasets.append(_load_single_dataset(dataset_attr, data_args))

    if len(datasets) == 1:
        return datasets[0]
    else:
        return concatenate_datasets(datasets)
    
def split_dataset(dataset: Union[Dataset, IterableDataset], val_size: float, seed: int=DATASET_RANDOM_SEED) -> DatasetDict:
    r"""
    Splits the dataset and returns a dataset dict containing train set and validation set.

    Supports both map dataset and iterable dataset.
    """
    val_size = int(val_size) if val_size > 1 else val_size
    dataset = dataset.train_test_split(test_size=val_size, seed=seed)
    return DatasetDict({"train": dataset["train"], "validation": dataset["test"]})

def get_dataset(data_args: DataArguments) -> DatasetDict:
    r"""
    Gets the train dataset and optionally gets the evaluation dataset.
    """
    # Load and preprocess dataset
    dataset = merge_dataset(data_args.dataset, data_args)
    eval_dataset = merge_dataset(data_args.eval_dataset, data_args)
    
    dataset_dict = {}
    if data_args.val_size > 1e-6:
        dataset_dict = split_dataset(dataset, data_args.val_size, data_args.seed)
    else:
        if dataset is not None:
            dataset_dict["train"] = dataset

        if eval_dataset is not None:
            dataset_dict["validation"] = eval_dataset

        dataset_dict = DatasetDict(dataset_dict)

    return dataset_dict

def preprocess_dataset_based_on_modality(dataset_dict: DatasetDict, data_args: DataArguments, modality: str):
    modality_columns = []
    if modality == "text":
        modality_columns = [TEXT_SRC, TEXT_HYP, TEXT_REF]
    elif modality == "vision":
        modality_columns = [IMG_SRC, TEXT_SRC, TEXT_HYP, TEXT_REF]
    elif modality == "reward":
        modality_columns = [CHOSEN, REJECTED]
    else:
        raise NotImplementedError(f"Modality `{modality}` is not recognized!")
    
    # Filter each split in the DatasetDict (e.g., "train", "validation")
    filtered_dataset_dict = {}
    for split_name, dataset in dataset_dict.items():
        columns_to_remove = [col for col in dataset.column_names if col not in modality_columns]
        filtered_dataset_dict[split_name] = dataset.remove_columns(columns_to_remove)

    return DatasetDict(filtered_dataset_dict)

def parse_dataset_args(dataset_config_path: str, modality: str) -> DatasetDict:
    parser = HfArgumentParser(DataArguments)

    if dataset_config_path.endswith(".yaml") or dataset_config_path.endswith(".yml"):
        parsed_data_args = parser.parse_yaml_file(os.path.abspath(dataset_config_path))
    elif dataset_config_path.endswith(".json"):
        parsed_data_args = parser.parse_json_file(os.path.abspath(dataset_config_path))
    else:
        logger.error("Got invalid dataset config path: {}".format(dataset_config_path))
        raise ValueError("dataset config path should be either JSON or YAML but got {} instead".format(dataset_config_path))

    dataset_dict = get_dataset(parsed_data_args)
    
    return preprocess_dataset_based_on_modality(dataset_dict, parsed_data_args, modality)
