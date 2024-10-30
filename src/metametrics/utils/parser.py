import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class MainArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    modality: str = field(
        metadata={"help": "The modality used for MetaMetrics."},
    )
    output_dir: str = field(
        metadata={"help": "The output directory of the experiments."},
    )
    optimizer_config_path: str = field(
        metadata={"help": "YAML/JSON file that contains optimizer config to be used for MetaMetrics."},
    )
    metrics_config_path: str = field(
        metadata={"help": "YAML/JSON file that contains list of metrics to be used for MetaMetrics."},
    )
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
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

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

def get_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(MainArguments)
    
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)

    