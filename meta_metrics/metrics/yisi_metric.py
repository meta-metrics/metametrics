import os
import gdown
import gzip
import shutil
import logging

from typing import List, Union
from .base_metric import BaseMetric

class YiSiMetric(BaseMetric):
    """
        args:
            metric_args (Dict): a dictionary of metric arguments
    """
    def __init__(self, **kwargs):
        self.l1 = 'en'
        self.l2 = 'en'
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.yisi_home = os.path.join(cur_dir, "yisi")
        self.yisi_model = os.path.join(self.yisi_home, "w2v_model.bin")
        if not os.path.exists(self.yisi_model):
            # Download Word2Vec Model
            gzip_file = os.path.join(self.yisi_home, "w2v_model.bin.gz")
            logging.info("Word2Vec Model not Found")
            gdown.download("https://drive.usercontent.google.com/open?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM", gzip_file)
            with gzip.open(gzip_file, 'rb') as f_in:
                with open(self.yisi_model, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(gzip_file)
        self.yisi_bin = os.path.join(self.yisi_home, 'bin/yisi')
        self.temp_folder = '/tmp/yisitmp'
        os.makedirs(self.temp_folder, exist_ok=True)
        

    def score(self, predictions: List[str], references: List[str], sources: Union[None, List[List[str]]]=None) -> List[float]:
        temp_file_ref_path = os.path.join(self.temp_folder, "temp_ref.en")
        temp_file_hyp_path = os.path.join(self.temp_folder, "temp_hyp.en")
        temp_file_sntscore_path = os.path.join(self.temp_folder, "temp_hyp.sntyisi1")
        temp_file_docscore_path = os.path.join(self.temp_folder, "temp_hyp.docyisi1")
        temp_file_config = os.path.join(self.temp_folder, "yisi.config")
        with open(temp_file_ref_path, 'w+') as f:
            for sent in references:
                f.write(sent + '\n')
        with open(temp_file_hyp_path, 'w+') as f:
            for sent in predictions:
                f.write(sent + '\n')
        
        cfg_string = f"""srclang={self.l1}
tgtlang={self.l2}
lexsim-type=w2v
outlexsim-path={self.yisi_model}
reflexweight-type=learn
phrasesim-type=nwpr
ngram-size=3
mode=yisi
alpha=0.8
ref-file={temp_file_ref_path}
hyp-file={temp_file_hyp_path}
sntscore-file={temp_file_sntscore_path}
docscore-file={temp_file_docscore_path}"""
        with open(temp_file_config, 'w+') as f:
            f.write(cfg_string)
        status = os.system(f"{self.yisi_bin} --config {temp_file_config}")
        if status != 0:
            raise ValueError('yisi failed to run')
        else:
            with open(temp_file_sntscore_path, 'r') as f:
                scores = [float(line.strip()) for line in  f.readlines()]
                return scores