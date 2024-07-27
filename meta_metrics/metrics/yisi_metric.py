from typing import List, Union, Dict, Tuple
from .base_metric import BaseMetric
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class YiSiGraph:
    def __init__(self, ref: str, hyp: str, ref_embedding, hyp_embedding):
        self.ref = ref
        self.hyp = hyp
        
        self.ref_embedding = ref_embedding
        self.hyp_embedding = hyp_embedding

    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    def get_token_similarities(self) -> torch.Tensor:
        # Compute the cosine similarity between each token in reference and hypothesis
        similarities = torch.matmul(self.ref_embedding, self.hyp_embedding.T)
        similarities = similarities / (torch.norm(self.ref_embedding, dim=1).unsqueeze(1) * torch.norm(self.hyp_embedding, dim=1))
        return similarities

class YiSiMetric(BaseMetric):
    def __init__(self, model_name: str = 'bert-base-multilingual-cased', alpha: float = 0.8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.alpha = alpha

    def _get_token_embeddings(self, text: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state  # shape (no_sentence, sequence_length, hidden_size)
        return token_embeddings

    def _compute_idf(self, documents: List[str]) -> Dict[str, float]:
        idf = {}
        total_docs = len(documents)
        all_words = set(word for sentence in documents for word in self.tokenizer.tokenize(sentence))
        for word in all_words:
            count = sum(1 for sentence in documents if word in self.tokenizer.tokenize(sentence))
            idf[word] = np.log((total_docs + 1) / (count + 1)) + 1
        return idf

    def _compute_features(self, 
                          prediction: str, 
                          reference: str,
                          prediction_embedding, 
                          reference_embedding,
                          idf_weights: Dict[str, float]) -> Tuple[float, float]:
        yisigraph = YiSiGraph(reference, prediction, reference_embedding, prediction_embedding)
        token_similarities = yisigraph.get_token_similarities()

        precision = 0.0
        recall = 0.0

        pred_tokens = self.tokenizer.tokenize(prediction)
        ref_tokens = self.tokenizer.tokenize(reference)
        
        weight_pred = sum(idf_weights.get(token, 0) for token in pred_tokens)
        weight_ref = sum(idf_weights.get(token, 0) for token in ref_tokens)

        # Compute precision
        for i, pred_token in enumerate(pred_tokens):
            max_weighted_sim = max(token_similarities[:, i] * idf_weights.get(pred_token, 0))
            precision += max_weighted_sim
        
        # Compute recall
        for j, ref_token in enumerate(ref_tokens):
            max_weighted_sim = max(token_similarities[j, :] * idf_weights.get(ref_token, 0))
            recall += max_weighted_sim

        precision /= weight_pred if weight_pred > 0 else 1
        recall /= weight_ref if weight_ref > 0 else 1
        
        return precision, recall

    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        idf_weights = self._compute_idf(predictions + references)
        pred_embeddings = self._get_token_embeddings(predictions)
        ref_embeddings = self._get_token_embeddings(references)
        
        scores = []
        for pred, ref, pred_e, ref_e in zip(predictions, references, pred_embeddings, ref_embeddings):
            precision, recall = self._compute_features(pred, ref, pred_e, ref_e, idf_weights)
            
            score = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
            scores.append(score.item())
        
        return scores
