from torchmetrics.functional.multimodal import clip_score
from functools import partial

import torch
import os
import logging
import requests
from zipfile import ZipFile
from typing import List, Union

from bleurt import score

from meta_metrics.metrics.base_metric import VisionToTextBaseMetric

class ClipScoreMetric(VisionToTextBaseMetric):
    """

    """
    def __init__(self, model_name: str, nthreads: int=16, **kwargs):
        self.clip_score_fn = partial(clip_score, model_name_or_path=model_name)

    def score(self, image_sources: List[torch.Tensor], text_predictions: List[str], text_references: Union[None, List[str]]=None, text_sources: Union[None, List[str]]=None) -> List[float]:
        images_int = (image_sources * 255).astype("uint8")
        clip_score = self.clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        return round(float(clip_score), 4)