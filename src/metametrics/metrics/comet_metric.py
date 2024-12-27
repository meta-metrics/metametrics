import os
from typing import List, Union, Optional
from pathlib import Path

from comet import load_from_checkpoint
from huggingface_hub import snapshot_download
import numpy as np

from metametrics.metrics.base_metric import TextBaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

from metametrics.utils.logging import get_logger
from metametrics.utils.constants import HF_TOKEN

logger = get_logger(__name__)

class COMETMetric(TextBaseMetric):
    def __init__(self, comet_model: str="Unbabel/XCOMET-XXL", batch_size: int=8, gpus: int=1,
                 reference_free: bool=False, hf_token: str=HF_TOKEN, **kwargs):
        # Choose model from the model hub
        self.comet_model = comet_model
        self.batch_size = validate_int(batch_size, valid_min=1)
        self.gpus = validate_int(gpus, valid_min=0)
        self.reference_free = validate_bool(reference_free)
        self.hf_token = hf_token
        if self.hf_token == "":
            logger.warning("HuggingFace Token is not filled, this may cause issues when downloading the model!")

    # Copy from comet's download_model to include hf_token
    def download_model(self, model: str, saving_directory: Union[str, Path, None]=None,
                       local_files_only: bool=False) -> str:
        try:
            model_path = snapshot_download(
                repo_id=model, cache_dir=saving_directory, local_files_only=local_files_only, token=self.hf_token,
            )
        except Exception:
            raise KeyError(f"COMET failed to download model '{model}'.")
        else:
            checkpoint_path = os.path.join(*[model_path, "checkpoints", "model.ckpt"])
        return checkpoint_path
    
    def _initialize_model(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = self.download_model(self.comet_model, saving_directory=os.path.join(cur_dir, "comet_models", self.comet_model))
        self.model = load_from_checkpoint(model_path)

    def score(self, predictions: List[str], references: Union[None, List[str]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        self._initialize_model()
        
        # Data must be converted into the following format for COMET
        if self.reference_free:
            data = [{"src": src, "mt": mt} for src, mt in zip(sources, predictions)]
        else:
            data = [{"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(sources, predictions, references)]
        return self.model.predict(data, batch_size=self.batch_size, gpus=self.gpus).scores

    @property
    def min_val(self) -> Optional[float]:
        return 0.0

    @property
    def max_val(self) -> Optional[float]:
        return 1.0

    @property
    def higher_is_better(self) -> bool:
        """Indicates if a higher value is better for this metric."""
        return True

    def __eq__(self, other):
        if isinstance(other, COMETMetric):
            self_vars = {k: v for k, v in vars(self).items() if k not in ['model', 'gpus']}
            other_vars = {k: v for k, v in vars(other).items() if k not in ['model', 'gpus']}
        
            return self_vars == other_vars
 
        return False
