import os
import joblib
import sys
import json
import yaml
from typing import Any, Dict, Optional

import pandas as pd
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datasets import Dataset

from metametrics.tasks.base import MetaMetrics
from metametrics.tasks.text import MetaMetricsText
from metametrics.tasks.vision import MetaMetricsVision
from metametrics.tasks.reward import MetaMetricsReward
from metametrics.tasks.evaluation.wmt_eval import evaluate_wmt23, evaluate_wmt24
from metametrics.tasks.evaluation.rewardbench_eval import evaluate_rewardbench

from metametrics.utils.logging import get_logger
from metametrics.utils.loader import parse_dataset_args, resolve_path
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
    evaluation_method: Optional[str] = field(
        default=None,
        metadata={"help": "The evaluation method of the experiments."},
    )
    evaluation_only: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to do only evaluation using an optimized MetaMetrics. Must provide optimizer_load_path if this is set to be True."},
    )
    pipeline_load_path: Optional[str] = field(
        default=None,
        metadata={"help": "Object file that contains pipeline to be used for MetaMetrics. Only use this if evaluation_only is True."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace token to access datasets (and potentially models)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for the datasets and models."}
    )
    normalize_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to normalize metrics used for MetaMetrics."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "Whether to overwrite the directory of the experiments."},
    )
    
    def __post_init__(self):
        # Check if output_dir exists and if overwrite_output_dir is False
        if os.path.exists(self.output_dir) and not self.overwrite_output_dir:
            raise FileExistsError(
                f"The output directory '{self.output_dir}' already exists and `overwrite_output_dir` is set to False."
            )
            
        if self.modality not in ["text", "vision", "reward"]:
            raise NotImplementedError(
                f"Modality of `{self.evaluation_method}` is not recognized!"
            )
            
        if self.evaluation_method is not None and self.evaluation_method not in ["wmt23", "wmt24", "rewardbench"]:
            raise NotImplementedError(
                f"Evaluation method of `{self.evaluation_method}` is not recognized!"
            )
            
        if self.evaluation_only == True and self.optimizer_load_path is None:
            raise ValueError(
                f"Evaluation only is set to be True, but optimizer_load_path is not provided!"
            )
            

def get_main_args(args: Optional[Dict[str, Any]] = None) -> MainArguments:
    parser = HfArgumentParser(MainArguments)
    
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and (sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")):
        return parser.parse_yaml_file(resolve_path(sys.argv[1]))[0]

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(resolve_path(sys.argv[1]))[0]

    parsed_args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        logger.warning(parser.format_help())
        logger.error("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return parsed_args

def parse_optimizer(optimizer_config_path: str):
    if optimizer_config_path.endswith(".yaml") or optimizer_config_path.endswith(".yml"):
        try:
            with open(resolve_path(optimizer_config_path), 'r') as f:
                parsed_optimizer = yaml.safe_load(f)
                return parsed_optimizer
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise ValueError("Failed to parse YAML configuration file.") from e
    elif optimizer_config_path.endswith(".json"):
        try:
            with open(resolve_path(optimizer_config_path), 'r') as f:
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
            with open(resolve_path(metrics_config_path), 'r') as f:
                parsed_metrics = yaml.safe_load(f)
                return parsed_metrics
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {e}")
            raise ValueError("Failed to parse YAML configuration file.") from e
    elif metrics_config_path.endswith(".json"):
        try:
            with open(resolve_path(metrics_config_path), 'r') as f:
                parsed_metrics = json.load(f)
                return parsed_metrics
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise ValueError("Failed to parse JSON configuration file.") from e
    else:
        logger.error("Got invalid config path: {}".format(metrics_config_path))
        raise ValueError("Config path should be either JSON or YAML but got {} instead".format(metrics_config_path))

def run_evaluate_metametrics(main_args: MainArguments,
                             task_pipeline: MetaMetrics,
                             eval_dataset: Dataset
                            ) -> None:
    # Initialize some variables
    normalize_metrics = main_args.normalize_metrics
    output_dir = resolve_path(main_args.output_dir)
    
    # Evaluate task
    logger.info("Evaluating metrics for validation dataset!")
    eval_metric_scores = task_pipeline.evaluate_metrics(eval_dataset, normalize_metrics)
    
    # Repeat the process for validation scores with unique names
    eval_scores_dict = {}
    unique_names = {}
    for name, scores in zip([metric.metric_name for metric in task_pipeline.get_metrics()], eval_metric_scores):
        if name in unique_names:
            unique_name = f"{name}_{unique_names[name]}"
        else:
            unique_name = name
            unique_names[name] = 1
            
        unique_names[name] += 1
        eval_scores_dict[unique_name] = scores
    
    eval_scores_df = pd.DataFrame(eval_scores_dict)
    
    logger.info("Predicting metametrics for evaluation dataset!")
    eval_pred = task_pipeline.predict_metametrics(eval_scores_df)
    eval_scores_df[consts.METAMETRICS_SCORE] = eval_pred
    
    # Save eval_scores_df
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    eval_scores_path = os.path.join(output_dir, "eval_scores.csv")
    eval_scores_df.to_csv(eval_scores_path, index=False)
    
    # Evaluate differently
    if main_args.evaluation_method is None:
        result = task_pipeline.evaluate_metametrics(eval_pred, eval_dataset[consts.TARGET_SCORE])
        # Save results
        result_path = os.path.join(output_dir, "result.csv")
        if isinstance(result, dict):
            # Single dictionary
            pd.DataFrame([result]).to_csv(result_path, index=False)
        elif isinstance(result, list):
            # List of dictionaries or simple values
            if all(isinstance(r, dict) for r in result):
                pd.DataFrame(result).to_csv(result_path, index=False)
            else:
                pd.DataFrame(result, columns=["result"]).to_csv(result_path, index=False)
        else:
            # Any other structure
            pd.DataFrame([result], columns=["result"]).to_csv(result_path, index=False)
    elif main_args.evaluation_method == "wmt23":
        evaluate_wmt23(eval_scores_df, output_dir)
    elif main_args.evaluation_method == "wmt24":
        evaluate_wmt24(eval_scores_df, output_dir)
    elif main_args.evaluation_method == "rewardbench":
        evaluate_rewardbench(eval_scores_df, output_dir)
    else:
        raise RuntimeError(
            f"Invalid evaluation method. This should've been checked earlier, unknown error!"
        )

def run_metametrics(args: Optional[Dict[str, Any]] = None) -> None:
    main_args = get_main_args(args)
    
    # Set some environment variables for HF Token and Cache Dir
    if main_args.hf_hub_token:
        consts.HF_TOKEN = main_args.hf_hub_token

    if main_args.cache_dir:
        consts.CACHE_DIR = main_args.cache_dir
    
    task_pipeline = None
    dataset_dict = parse_dataset_args(main_args.dataset_config_path, modality=main_args.modality)
    output_dir = resolve_path(main_args.output_dir)
    if main_args.evaluation_only:
        try:
            task_pipeline = joblib.load(open(main_args.pipeline_load_path, "rb"))
        except (FileNotFoundError, EOFError, ValueError) as e:
            raise RuntimeError(f"Failed to load pipeline: {e}")
    else:
        # Parse config files
        optimizer = parse_optimizer(main_args.optimizer_config_path)
        metrics_list = parse_metrics_dict(main_args.metrics_config_path)
        normalize_metrics = main_args.normalize_metrics

        # Initialize pipeline
        if main_args.modality == "text":
            task_pipeline = MetaMetricsText()
        elif main_args.modality == "vision":
            task_pipeline = MetaMetricsVision()
        elif main_args.modality == "reward":
            task_pipeline = MetaMetricsReward()
        else:
            raise RuntimeError(
                f"Invalid modality. This should've been checked earlier, unknown error!"
            )

        # Add metrics
        for metric in metrics_list:
            task_pipeline.add_metric(metric.get("metric_name"), metric.get("metric_args"))
        
        # Set optimizer
        task_pipeline.set_optimizer(optimizer.get("optimizer_name"), optimizer.get("optimizer_args"))
        
        # Save train dataset
        os.makedirs(output_dir, exist_ok=True)
        train_dataset_path = os.path.join(output_dir, "train_dataset.pkl")
        joblib.dump(dataset_dict["train"], train_dataset_path, compress=("gzip", 3))
        
        # Evaluate train metrics
        logger.info("Evaluating metrics for train dataset!")
        train_metric_scores = task_pipeline.evaluate_metrics(dataset_dict["train"], normalize_metrics)
        
        # Create unique metric names for the DataFrame
        unique_names = {}
        train_scores_dict = {}
        
        for name, scores in zip([metric.metric_name for metric in task_pipeline.get_metrics()], train_metric_scores):
            # Create a unique name if duplicate
            if name in unique_names:
                unique_name = f"{name}_{unique_names[name]}"
            else:
                unique_name = name
                unique_names[name] = 1

            unique_names[name] += 1
            train_scores_dict[unique_name] = scores
            
        # Convert to DataFrame
        train_scores_df = pd.DataFrame(train_scores_dict)
        
        # Calibrate
        logger.info("Calibrate and predict metametrics based on train dataset!")
        task_pipeline.calibrate(train_scores_df, dataset_dict["train"])
        train_pred = task_pipeline.predict_metametrics(train_scores_df)
        train_scores_df[consts.METAMETRICS_SCORE] = train_pred
        
        # Save train_scores_df
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        train_scores_path = os.path.join(output_dir, "train_scores.csv")
        train_scores_df.to_csv(train_scores_path, index=False)
    
    if "validation" in dataset_dict:
        eval_dataset_path = os.path.join(output_dir, "eval_dataset.pkl")
        joblib.dump(dataset_dict["validation"], eval_dataset_path, compress=("gzip", 3))
        run_evaluate_metametrics(main_args, task_pipeline, dataset_dict["validation"])
    
    # Save task pipeline
    task_pipeline_save_path = os.path.join(output_dir, "task_pipeline.pkl")
    joblib.dump(task_pipeline, task_pipeline_save_path, compress=("gzip", 3))

    logger.info("Finished running MetaMetrics!")
