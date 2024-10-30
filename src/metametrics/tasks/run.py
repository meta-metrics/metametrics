import os
import shutil
from typing import Any, Dict, List, Optional

import torch

from metametrics.utils.logging import get_logger
from metametrics.utils.parser import get_args

logger = get_logger(__name__)


def run_metametrics(args: Optional[Dict[str, Any]] = None) -> None:
    optimizer_args, data_args, metric_args = get_train_args(args)

    if finetuning_args.stage == "text":
        run_pt(optimizer_args, data_args, metric_args)
    elif finetuning_args.stage == "vision":
        run_sft(optimizer_args, data_args, metric_args)
    elif finetuning_args.stage == "reward":
        run_rm(optimizer_args, data_args, metric_args)
    else:
        raise ValueError("Unknown task: {}.".format(finetuning_args.stage))
