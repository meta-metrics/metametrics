from typing import List, Union, Optional
import logging
import tempfile
import os
import shutil
import numpy as np
from pyrouge import Rouge155

from metametrics.metrics.base_metric import TextBaseMetric
from metametrics.utils.validate import validate_argument_list, validate_int, validate_real, validate_bool

from metametrics.utils.logging import get_logger

logger = get_logger(__name__)

class ROUGEMetric(TextBaseMetric):
    def __init__(self, rouge_type="rouge1", rouge_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "ROUGE-1.5.5"), **kwargs):
        if not os.path.isdir(rouge_dir) and not os.path.isdir(os.environ['ROUGE_HOME']):
            raise FileNotFoundError("ROUGE HOME is not found. Hint: do `pip install \".[rouge]\"`")
        
        self.rouge_dir = rouge_dir
        self.rouge_type = rouge_type
        
    def _initialize_metric(self):
        self.r = Rouge155(rouge_dir=self.rouge_dir, rouge_args=None, log_level=logging.ERROR)

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        self._initialize_metric()
        
        segment_scores = []
        
        for pred, refs in zip(predictions, references):
            self.r.system_dir = tempfile.mkdtemp()
            self.r.model_dir = tempfile.mkdtemp()
            self.r.system_filename_pattern = 'system.(\d+).txt'
            self.r.model_filename_pattern = 'model.[A-Z].#ID#.txt'
            with open(os.path.join(self.r.system_dir, "system.0.txt"), "w") as outputf:
                outputf.write(pred)
            for ref_idx, ref in enumerate(refs):
                with open(os.path.join(self.r.model_dir, f"model.{chr(ord('A') + ref_idx)}.0.txt"), "w") as outputf:
                    outputf.write(ref)
            
            output = self.r.convert_and_evaluate()
            output_dict = self.r.output_to_dict(output)
            shutil.rmtree(self.r.system_dir)
            shutil.rmtree(self.r.model_dir)
            segment_scores.append(output_dict[self.rouge_type])

        return segment_scores
    
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
        if isinstance(other, ROUGEMetric):
            self_vars = {k: v for k, v in vars(self).items() if k not in ['r']}
            other_vars = {k: v for k, v in vars(other).items() if k not in ['r']}
        
            return self_vars == other_vars
 
        return False
