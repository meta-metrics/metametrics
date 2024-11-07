import os
import sys
import json
import yaml
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from metametrics.tasks.text import run_metametrics_text
from metametrics.tasks.vision import run_metametrics_vision
from metametrics.tasks.reward import run_metametrics_reward

from metametrics.utils.logging import get_logger
from metametrics.utils.loader import parse_dataset_args
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
    normalize_metrics: bool = field(
        default=True,
        metadata={"help": "Normalize metrics used for MetaMetrics."},
    )
    output_dir: str = field(
        metadata={"help": "The output directory of the experiments."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Whether to overwrite the directory of the experiments."},
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

def parse_optimizer(optimizer_config_path: str):
    if optimizer_config_path.endswith(".yaml") or optimizer_config_path.endswith(".yml"):
        try:
            with open(os.path.abspath(optimizer_config_path), 'r') as f:
                parsed_optimizer = yaml.safe_load(f)
                return parsed_optimizer
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise ValueError("Failed to parse YAML configuration file.") from e
    elif optimizer_config_path.endswith(".json"):
        try:
            with open(os.path.abspath(optimizer_config_path), 'r') as f:
                parsed_optimizer = json.load(f)
                return parsed_optimizer
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise ValueError("Failed to parse JSON configuration file.") from e
    else:
        logger.error("Got invalid config path: {}".format(optimizer_config_path))
        raise ValueError("Config path should be either JSON or YAML but got {} instead".format(optimizer_config_path))

def parse_metrics_dict(metrics_config_path: str):
    if metrics_config_path.endswith(".yaml") or metrics_config_path.endswith(".yml"):
        try:
            with open(os.path.abspath(metrics_config_path), 'r') as f:
                parsed_metrics = yaml.safe_load(f)
                return parsed_metrics
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise ValueError("Failed to parse YAML configuration file.") from e
    elif metrics_config_path.endswith(".json"):
        try:
            with open(os.path.abspath(metrics_config_path), 'r') as f:
                parsed_metrics = json.load(f)
                return parsed_metrics
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise ValueError("Failed to parse JSON configuration file.") from e
    else:
        logger.error("Got invalid config path: {}".format(metrics_config_path))
        raise ValueError("Config path should be either JSON or YAML but got {} instead".format(metrics_config_path))

def run_metametrics(args: Optional[Dict[str, Any]] = None) -> None:
    main_args = get_main_args(args)
    
    # Set some environment variables for HF Token and Cache Dir
    if main_args.hf_hub_token:
        consts.HF_TOKEN = main_args.hf_hub_token

    if main_args.cache_dir:
        consts.CACHE_DIR = main_args.cache_dir
        
    # Parse config files
    optimizer = parse_optimizer(main_args.optimizer_config_path)
    dataset_dict = parse_dataset_args(main_args.dataset_config_path, modality=main_args.modality)
    metrics_list = parse_metrics_dict(main_args.metrics_config_path)

    if main_args.modality == "text":
        run_metametrics_text(optimizer, dataset_dict, metrics_list,
                             main_args.normalize_metrics, main_args.output_dir, main_args.overwrite_output_dir)
    elif main_args.modality == "vision":
        run_metametrics_vision(optimizer, dataset_dict, metrics_list)
    elif main_args.modality == "reward":
        run_metametrics_reward(optimizer, dataset_dict, metrics_list)
    else:
        raise NotImplementedError(f"Modality `{main_args.modality}` is not recognized!")
