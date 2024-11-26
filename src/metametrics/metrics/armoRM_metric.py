import pandas as pd
import json
from typing import Dict, List, Union
from meta_metrics.metrics.base_metric import BaseMetric
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class ArmoRMMetric(BaseMetric):
    class ArmoRMPipeline:
        def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
            )
            self.truncation = truncation
            self.device = self.model.device
            self.max_length = max_length
            self.attributes = [
                'helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
                'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
                'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
                'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
                'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
                'code-style','code-explanation','code-instruction-following','code-readability'
            ]
            self.attribute_dict = {k: v for (v, k) in enumerate(self.attributes)}
    
        def __call__(self, messages: List[Dict[str, str]], scoring_attribute: str) -> Dict[str, float]:
            """
            messages: OpenAI chat messages to be scored
            scoring_attribute: attribute for scoring, use depending on cases.
            Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
            Returns: a dictionary with the score between 0 and 1
            """

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                padding=True,
                truncation=self.truncation,
                max_length=self.max_length,
            ).to(self.device)
            with torch.no_grad():
               output = self.model(input_ids)
               multi_obj_rewards = output.rewards.cpu().float() 
               gating_output = output.gating_output.cpu().float()
               preference_score = output.score.cpu().float() 
            obj_transform = self.model.reward_transform_matrix.data.cpu().float()
            multi_obj_coeffs = gating_output @ obj_transform.T
            assert torch.isclose(torch.sum(multi_obj_rewards * multi_obj_coeffs, dim=1), preference_score, atol=1e-3) 
            return {scoring_attribute: multi_obj_rewards[0][self.attribute_dict[scoring_attribute]]}

    def __init__(self, sources=None, predictions=None, scoring_attribute=None):
        # In the context of QA, sources are the question and predictions are the model output answer.
        self.rm = self.ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)
        self.sources = sources
        self.predictions = predictions
        self.scoring_attribute = scoring_attribute
        if scoring_attribute == None:
            self.scoring_attribute = self.rm.attributes[1]
        
    def score(self, predictions: List[str]=None, references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        if self.predictions is None:
            self.predictions = predictions
        if self.sources is None:
            self.sources = sources
        df = pd.DataFrame({
            'question': sources,
            'prediction': predictions
        })
        df['id'] = range(len(predictions))
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                score = self.rm(
                    messages = [{"role": "user", "content": row['question']}, {"role": "assistant", "content": row['prediction']}],
                    scoring_attribute = self.scoring_attribute
                )
                scores.append(score[self.scoring_attribute])
            except Exception as e:
                print(f"An error occured: {e} || Setting default error value to -1")
                scores.append(-1)
        return scores
