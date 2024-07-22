from typing import List, Union, Dict, Tuple
from .base_metric import BaseMetric
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class YiSiGraph:
    def __init__(self, ref: str, hyp: str, model_name: str = 'bert-base-multilingual-cased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        self.ref = ref
        self.hyp = hyp
        
        self.ref_embedding = self._get_token_embeddings(ref)
        self.hyp_embedding = self._get_token_embeddings(hyp)

    def _get_token_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state.squeeze(0)  # shape (sequence_length, hidden_size)
        return token_embeddings

    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    def get_token_similarities(self) -> List[List[float]]:
        similarities = []
        for ref_emb in self.ref_embedding:
            ref_sims = []
            for hyp_emb in self.hyp_embedding:
                sim = self._cosine_similarity(ref_emb, hyp_emb)
                ref_sims.append(sim)
            similarities.append(ref_sims)
        return similarities

class YiSiMetric(BaseMetric):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', alpha: float = 0.8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        self.alpha = alpha

    def _compute_idf(self, documents: List[str]) -> Dict[str, float]:
        idf = {}
        total_docs = len(documents)
        all_words = set(word for sentence in documents for word in self.tokenizer.tokenize(sentence))
        for word in all_words:
            count = sum(1 for sentence in documents if word in self.tokenizer.tokenize(sentence))
            idf[word] = np.log((total_docs + 1) / (count + 1)) + 1
        return idf

    def _compute_features(self, prediction: str, reference: str, idf_weights: Dict[str, float]) -> Tuple[float, float]:
        yisigraph = YiSiGraph(reference, prediction, self.model_name)
        token_similarities = yisigraph.get_token_similarities()

        precision = 0.0
        recall = 0.0

        pred_tokens = self.tokenizer.tokenize(prediction)
        ref_tokens = self.tokenizer.tokenize(reference)
        
        weight_pred = 0
        weight_ref = 0

        # Compute precision
        for i, pred_token in enumerate(pred_tokens):
            max_weighted_sim = 0
            for j, ref_token in enumerate(ref_tokens):
                weighted_sim = token_similarities[j][i] * idf_weights.get(pred_token, 0)
                if weighted_sim > max_weighted_sim:
                    max_weighted_sim = weighted_sim
            weight_pred += idf_weights.get(pred_token, 0)
            precision += max_weighted_sim
        
        # Compute recall
        for j, ref_token in enumerate(ref_tokens):
            max_weighted_sim = 0
            for i, pred_token in enumerate(pred_tokens):
                weighted_sim = token_similarities[j][i] * idf_weights.get(ref_token, 0)
                if weighted_sim > max_weighted_sim:
                    max_weighted_sim = weighted_sim
            weight_ref += idf_weights.get(ref_token, 0)
            recall += max_weighted_sim
        
        precision /= weight_pred if weight_pred > 0 else 1
        recall /= weight_ref if weight_ref > 0 else 1
        
        return precision, recall

    def score(self, predictions: List[str], references: List[str], sources: Union[None, List[str]]=None) -> List[float]:
        idf_weights = self._compute_idf(predictions + references)
        
        scores = []
        for pred, ref in zip(predictions, references):
            precision, recall = self._compute_features(pred, ref, idf_weights)
            
            score = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
            scores.append(score)
        
        return scores