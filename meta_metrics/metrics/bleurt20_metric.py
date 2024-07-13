import os
import logging
import requests
from zipfile import ZipFile
from typing import List, Union

from bleurt import score

from .base_metric import BaseMetric

class BLEURT20Metric(BaseMetric):
    def __init__(self, **kwargs):
        # The paths
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        bleurt_folder_path = os.path.join(cur_dir, "bleurt")
        bleurt_zip_path = os.path.join(cur_dir, "bleurt/BLEURT-20.zip")
        bleurt_model_path = os.path.join(cur_dir, "bleurt/BLEURT-20")
        os.makedirs(bleurt_folder_path, exist_ok=True)
        
        # Check if BLEURT-20 is already downloaded
        if not os.path.exists(bleurt_model_path):
            # Download BLEURT-20
            logging.info("BLEURT-20 not found. Downloading and extracting...")
            with requests.get("https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip", stream=True) as r:
                r.raise_for_status()
                with open(bleurt_zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Unzip      
            with ZipFile(bleurt_zip_path, 'r') as zip_ref:
                zip_ref.extractall(bleurt_folder_path)
            os.remove(bleurt_zip_path)
        
        self.scorer = score.BleurtScorer(bleurt_model_path)

    def score(self, predictions: List[str], references: List[str], sources: Union[None, List[str]]=None) -> List[float]:
        return self.scorer.score(references=references, candidates=predictions)
