import os
import sys
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from metametrics.utils.logging import get_logger
import metametrics.utils.constants as consts

logger = get_logger(__name__)

@dataclass
class MainArguments:
    r"""
    Arguments pertaining to which MetaMetrics pipeline (optimizer, metrics, dataset) to be used.
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
    dataset_config_path: str = field(
        metadata={"help": "YAML/JSON file that contains dataset config to be used for MetaMetrics."},
    )
    hf_hub_token: Optional[str] = field(
        metadata={"help": "HuggingFace token to access datasets (and potentially models)."},
    )
    cache_dir: Optional[str] = field(
        metadata={"help": "Cache directory for the datasets and models."}
    )

def get_main_args(args: Optional[Dict[str, Any]] = None) -> MainArguments:
    parser = HfArgumentParser(MainArguments)
    
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    parsed_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        logger.warning(parser.format_help())
        logger.error("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return parsed_args


def run_metametrics(args: Optional[Dict[str, Any]] = None) -> None:
    main_args = get_main_args(args)
    
    # Set some environment variables for HF Token and Cache Dir
    if main_args.hf_hub_token:
        consts.HF_TOKEN = main_args.hf_hub_token

    if main_args.cache_dir:
        consts.CACHE_DIR = main_args.cache_dir

    if main_args.modality == "text":
        run_pt(optimizer_args, data_args, metric_args)
    elif main_args.modality == "vision":
        run_sft(optimizer_args, data_args, metric_args)
    elif main_args.modality == "reward":
        run_rm(optimizer_args, data_args, metric_args)
    else:
        raise NotImplementedError(f"Modality `{modality}` is not recognized!")
