from typing import List, Dict, Union
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import evaluate
from transformers import AutoTokenizer, AutoModel
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from sacrebleu.metrics import BLEU
import torch
from dataclasses import dataclass
import warnings
from transformers import logging

# transformers 경고 끄기
logging.set_verbosity_error()

# 모든 경고 메시지 끄기
warnings.filterwarnings('ignore')

try:
    nltk.download('punkt')
except:
    pass

@dataclass
class MetricScores:
    bleu: float
    rouge_1: float
    rouge_2: float
    rouge_l: float
    meteor: float
    bert_score: float


class TextDataset(Dataset):
    def __init__(self, predictions: List[List[str]], references: List[str]):
        assert len(predictions) == len(references)
        self.predictions = predictions  # List[List[str]]
        self.references = references    # List[str]
        
    def __len__(self):
        return len(self.references)
    
    def __getitem__(self, idx):
        return self.predictions[idx], self.references[idx]  # Returns (List[str], str)


class Similarity:
    def __init__(self, batch_size: int = 80, use_gpu: bool = True):
        self.batch_size = batch_size
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU(effective_order=True)
        
    def compute_bert_score(self, predictions: List[str], references: List[str]) -> np.ndarray:
        P, R, F1 = score(predictions, references, lang='en', 
                        batch_size=self.batch_size, device=self.device)
        return F1.numpy()

    def compute_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(prediction, reference)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def compute_meteor_score(self, prediction: str, reference: str) -> float:
        return meteor_score([reference.split()], prediction.split())

    def compute_bleu_score(self, prediction: str, reference: str) -> float:
        return self.bleu.sentence_score(prediction, [reference]).score / 100.0

    def evaluate(self, predictions: List[str], references: List[str], return_all: bool = False) -> List[MetricScores]:
        assert len(predictions) == len(references), "Predictions and references must have same length"
        results = {}
        for key in MetricScores.__annotations__.keys():
            results[key] = []
            
        dataset = TextDataset(predictions, references)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        for prediction, reference in tqdm(dataloader, desc="Computing metrics"):
            prediction = prediction[0]
            reference = reference[0]
            bleu = self.compute_bleu_score(prediction, reference)
            rouge_scores = self.compute_rouge_scores(prediction, reference)
            meteor = self.compute_meteor_score(prediction, reference)
            bert_score = self.compute_bert_score([prediction], [reference])[0]
            
            results['bleu'].append(bleu)
            results['rouge_1'].append(rouge_scores['rouge1'])
            results['rouge_2'].append(rouge_scores['rouge2'])
            results['rouge_l'].append(rouge_scores['rougeL'])
            results['meteor'].append(meteor)
            results['bert_score'].append(bert_score)
        
        if return_all:
            return results
        
        else:
            return {key: np.mean(value) for key, value in results.items()}
    
    
    def group_evaluate(self, predictions: List[List[str]], references: List[str], the_most_similar: int = 5) -> List[MetricScores]:
        assert len(predictions) == len(references), "Predictions and references must have same length"
        results = {key: [] for key in MetricScores.__annotations__.keys()}
        
        # 각 그룹의 시작 인덱스 계산
        start_indices = [0]
        for group in predictions[:-1]:
            start_indices.append(start_indices[-1] + len(group))
        
        # 모든 예측에 대한 점수 계산
        flattened_predictions = [p for group in predictions for p in group]
        flattened_references = []
        for i, ref in enumerate(references):
            flattened_references.extend([ref] * len(predictions[i]))
        
        all_scores = self.evaluate(flattened_predictions, flattened_references, return_all=True)
        
        # 그룹별로 처리
        for i in range(len(references)):
            start_idx = start_indices[i]
            end_idx = start_indices[i] + len(predictions[i])
            
            # 현재 그룹의 BERTScore
            group_bert_scores = all_scores['bert_score'][start_idx:end_idx]
            
            # BERTScore 기준으로 상위 k개 인덱스 찾기
            k = min(the_most_similar, len(group_bert_scores))  # 그룹 크기가 k보다 작을 경우 처리
            top_k_indices = np.argsort(group_bert_scores)[-k:]
            
            # 각 메트릭에 대해 상위 k개 결과만 저장
            for key in results.keys():
                group_scores = all_scores[key][start_idx:end_idx]
                results[key].extend([group_scores[idx] for idx in top_k_indices])
        
        # 최종 평균 계산
        return {key: np.mean(value) for key, value in results.items()}