# MetaMetrics

The repository is the open-source implementation for MetaMetrics: Calibrating Metrics For Generation Tasks Using Human Preferences https://arxiv.org/pdf/2410.02381.
We will release the code soon.

## Contents

+ [Environment](#environment)
+ [Setup Instruction](#setup-instruction)
+ [Running Instruction](#running-instruction)

## Environment

Python 3.10 or higher. Details of dependencies are in `setup.py`.

## Setup Instruction

1. Run `pip install -e .` as it will automatically install required dependencies.
2. Optionally, you may install additional dependencies by running `pip install -e ".[<additional_dependencies>]"`.
Replace `<additional_dependencies>` with one (or more) of the following based on your needs:
- "wmt-eval": For WMT tasks.
- "rewardbench": For RewardBench tasks.
- "xgboost": For XGBoost optimizer.
- "gemba": For using GEMBA.

## Running Instruction

For concrete examples, you can look at some of the sample configs in `examples` folder.

Experiments can be run using the `metametrics-cli` command after installation. To run an experiment, simply provide a config file and execute the following command:

```bash
metametrics-cli run <path_to_config>
```

The configuration file should be in YAML/JSON format and must include the following arguments:

- `modality`: Modality for MetaMetrics (e.g., text).
- `output_dir`: Directory to save experiment results.
- `evaluation_method`: Method for evaluating experiments.
- `evaluation_only`: Set to True to perform evaluation only (requires `pipeline_load_path`).
- `pipeline_load_path`: Path to a pre-saved pipeline (used with `evaluation_only`).
- `optimizer_config_path`: Path to YAML/JSON with optimizer configuration.
- `metrics_config_path`: Path to YAML/JSON with metrics configuration.
- `dataset_config_path`: Path to YAML/JSON with dataset configuration.
- `hf_hub_token`: HuggingFace token for dataset access.
- `cache_dir`: Cache directory for datasets and models.
- `normalize_metrics`: Normalize metrics for MetaMetrics.
- `overwrite_output_dir`: Overwrite existing output directory.

More details of each argument can be found in `src/metametrics/tasks/run.py`. The next sections will discuss what configurations need to be provided for `dataset`, `metrics`, and `optimizer`.

### Datasets

`dataset_config_path` is a YAML/JSON file containing the configuration, with key fields such as:

- `train_dataset`: List of datasets used for training, defined with `DatasetAttr` attributes.
- `eval_dataset`: List of datasets used for evaluation, also defined with `DatasetAttr` attributes.
- `dataset_dir`: Path to the folder containing datasets.
- `preprocessing_num_workers`: Number of workers for preprocessing.
- `max_samples`: Limit the number of examples per dataset (useful for debugging).
- `val_size`: Proportion or number of samples for the validation set.
- `seed`: Seed for random operations on the dataset.

Each dataset listed in `dataset` or `eval_dataset` field have their attributes defined by the `DatasetAttr` class. They should have the following attributes:

- `load_from`: Source of the dataset (hf_hub for HuggingFace Hub or file for local files).
- `dataset_name`: Name of the dataset.
- `subset`: Subset of the dataset, if applicable.
- `split`: Dataset split to use (e.g., train, test).
- `folder`: Path to the folder containing the dataset (if load_from is file).

We can also define the appropriate column names:

Text Columns (optional):
- `text_src`: Column name that contains source text.
- `text_hyp`: Column name that contains hypothesis text.
- `text_ref`: Column name that contains reference text.

Vision Columns (optional):
- `img_src`: Column name that contains image source paths.

Target Column (optional):
- `target_score`: Column name that contains target scores for evaluation.

Detailed arguments for the dataset configuration can be found in `src/metametrics/utils/loader.py`

### Metrics

`metrics_config_path` is a YAML/JSON file containing list of metric names and their respective arguments. We allow same metrics with different arguments, but we exclude metrics with the same name and arguments for our MetaMetrics pipeline. You may also register new custom metrics.

Detailed arguments for the metrics can be found in `src/metametrics/metrics`.

### Optimizer

`optimizer_config_path` is a YAML/JSON file containing the configuration for the optimizer. You may also register new custom optimizers.

Detailed arguments for the metrics can be found in `src/metametrics/optimizer`.

## Citation

If you use this code for your research, please cite the following work:

```bibtex
@article{winata2024metametrics,
  title={Metametrics: Calibrating metrics for generation tasks using human preferences},
  author={Winata, Genta Indra and Anugraha, David and Susanto, Lucky and Kuwanto, Garry and Wijaya, Derry Tanti},
  journal={arXiv preprint arXiv:2410.02381},
  year={2024}
}
```

If you have any questions, you can open a [GitHub Issue](https://github.com/meta-metrics/metametrics/issues)!
