import os
import sys
import json
import yaml
from typing import Any, Dict, Optional

import pandas as pd
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from metametrics.tasks.text import MetaMetricsText
from metametrics.tasks.vision import MetaMetricsVision
from metametrics.tasks.reward import MetaMetricsReward

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
    
    def __post_init__(self):
        # Check if output_dir exists and if overwrite_output_dir is False
        if os.path.exists(self.output_dir) and not self.overwrite_output_dir:
            raise FileExistsError(
                f"The output directory '{self.output_dir}' already exists and `overwrite_output_dir` is set to False."
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
    normalize_metrics = main_args.normalize_metrics
    output_dir = os.path.abspath(main_args.output_dir)

    if main_args.modality == "text":
        task_pipeline = MetaMetricsText()
    elif main_args.modality == "vision":
        task_pipeline = MetaMetricsVision()
    elif main_args.modality == "reward":
        task_pipeline = MetaMetricsReward()
    else:
        raise NotImplementedError(f"Modality `{main_args.modality}` is not recognized!")
    
    
    # Add metrics
    for metric in metrics_list:
        task_pipeline.add_metric(metric.get("metric_name"), metric.get("metric_args"))
    
    # Set optimizer
    task_pipeline.set_optimizer(optimizer.get("optimizer_name"), optimizer.get("optimizer_args"))
    
    # Evaluate train metrics
    train_metric_scores = task_pipeline.evaluate_metrics(dataset_dict["train"], normalize_metrics)
    
    # Create unique metric names for the DataFrame
    unique_names = {}
    train_scores_dict = {}
    
    for name, scores in zip([metric["metric_name"] for metric in metrics_list], train_metric_scores):
        # Create a unique name if duplicate
        if unique_names[name] > 0:
            unique_name = f"{name}_{unique_names[name]}"
        else:
            unique_name = name
            unique_names[name] = 0

        unique_names[name] += 1
        train_scores_dict[unique_name] = scores
        
    # Convert to DataFrame
    train_scores_df = pd.DataFrame(train_scores_dict)
    
    # Save train_scores_df
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    train_scores_path = os.path.join(output_dir, "train_scores.csv")
    train_scores_df.to_csv(train_scores_path, index=False)

    # Calibrate
    task_pipeline.calibrate(train_scores_df, dataset_dict["train"]["target_scores"])
    
    # Evaluate task
    eval_metric_scores = task_pipeline.evaluate_metrics(dataset_dict["validation"], normalize_metrics)
    
    # Repeat the process for validation scores with unique names
    eval_scores_dict = {}
    unique_names.clear()  # Reset the counter for validation
    for name, scores in zip([m["metric_name"] for m in metrics_list], eval_metric_scores):
        if unique_names[name] > 0:
            unique_name = f"{name}_{unique_names[name]}"
        else:
            unique_name = name
        unique_names[name] += 1
        eval_scores_dict[unique_name] = scores
    
    eval_scores_df = pd.DataFrame(eval_scores_dict)
    
    # Save train_scores_df
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    eval_scores_path = os.path.join(output_dir, "eval_scores.csv")
    eval_scores_df.to_csv(eval_scores_path, index=False)
    
    pred, result = task_pipeline.evaluate_metametrics(eval_scores_df, dataset_dict["validation"]["target_scores"])
    
    # Save predictions
    pred_df = pd.DataFrame(pred, columns=["predictions"])
    human_scores_df = pd.DataFrame(dataset_dict["validation"]["target_scores"])
    pred_df = pd.concat([pred_df, human_scores_df], axis=1)
    pred_path = os.path.join(output_dir, "pred_human_scores.csv")
    pred_df.to_csv(pred_path, index=False)

    # Save results
    result_path = os.path.join(output_dir, "result.csv")
    if isinstance(result, dict):
        pd.DataFrame([result]).to_csv(result_path, index=False)
    else:
        pd.DataFrame(result).to_csv(result_path, index=False)
