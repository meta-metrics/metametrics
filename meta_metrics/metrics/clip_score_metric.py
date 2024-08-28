from torchmetrics.functional.multimodal import clip_score
from functools import partial

import torch
import os
import logging
import requests
import clip
from zipfile import ZipFile
from typing import List, Union
from tqdm import tqdm

import argparse
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import pathlib
import json
import generation_eval_utils
import pprint
import warnings

from packaging import version

from meta_metrics.metrics.base_metric import VisionToTextBaseMetric

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)

def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features

class ClipScoreMetric(VisionToTextBaseMetric):
    """

    """
    def __init__(self, is_reference_free:bool, model_name: str, device: str, w: float=2.5, **kwargs):
        # self.clip_score_fn = partial(clip_score, model_name_or_path=model_name)
        self.is_reference_free = is_reference_free
        self.model_name = model_name

        self.model, self.transform = clip.load("ViT-B/32", device=device, jit=False)
        self.model.eval()

        self.device = device
        self.w = w


    def score(self, image_sources: List[torch.Tensor], text_predictions: List[str], text_references: Union[None, List[str]]=None, text_sources: Union[None, List[str]]=None) -> List[float]:
        # prompts = []
        # for i in range(len(text_predictions)):
        #     prompts.append(text_predictions[i])
        
        # images_int = (image_sources * 255).astype("uint8")
        # clip_score = self.clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        # return round(float(clip_score), 4)
        if self.is_reference_free:
            if isinstance(candidates, list):
                candidates = extract_all_captions(candidates, self.model, self.device)

            flattened_refs = []
            flattened_refs_idxs = []
            for idx, refs in enumerate(references):
                flattened_refs.extend(refs)
                flattened_refs_idxs.extend([idx for _ in refs])

            flattened_refs = extract_all_captions(flattened_refs, self.model, self.device)

            if version.parse(np.__version__) < version.parse('1.21'):
                candidates = sklearn.preprocessing.normalize(candidates, axis=1)
                flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis=1)
            else:
                warnings.warn(
                    'due to a numerical instability, new numpy normalization is slightly different than paper results. '
                    'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

                candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
                flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

            cand_idx2refs = collections.defaultdict(list)
            for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
                cand_idx2refs[cand_idx].append(ref_feats)

            assert len(cand_idx2refs) == len(candidates)

            cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

            per = []
            for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
                cur_refs = cand_idx2refs[c_idx]
                all_sims = cand.dot(cur_refs.transpose())
                per.append(np.max(all_sims))

            return np.mean(per), per
        else:
            if isinstance(images, list):
                # need to extract image features
                images = extract_all_images(images, self.model, self.device)

                candidates = extract_all_captions(candidates, self.model, self.device)

                #as of numpy 1.21, normalize doesn't work properly for float16
                if version.parse(np.__version__) < version.parse('1.21'):
                    images = sklearn.preprocessing.normalize(images, axis=1)
                    candidates = sklearn.preprocessing.normalize(candidates, axis=1)
                else:
                    warnings.warn(
                        'due to a numerical instability, new numpy normalization is slightly different than paper results. '
                        'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
                    images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
                    candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

                per = self.w*np.clip(np.sum(images * candidates, axis=1), 0, None)
                return np.mean(per), per, candidates


