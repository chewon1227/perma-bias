from mylmeval import save_json, open_json, get_results
import fire, os, pandas as pd, re, evaluate
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm, trange
from bert_score import score
from dataclasses import dataclass
from sacrebleu.metrics import BLEU
import re
from get_score import TextEvaluator, make_gt, FILENAMES
from careerpathway.scoring import Diversity



def process_file_random_truthfulness(file_path: str, batch_size: int = 32, num_random_pairs: int = 1000):
    """Requirements와 Description에 대한 random baseline"""
    print("Computing random baseline for truthfulness...")
    evaluator = TextEvaluator(batch_size=batch_size)
    df = pd.read_json(file_path, lines=True)
    df['task'] = df['metadata'].apply(lambda x: x['task'])
    
    random_scores = {
        'requirement': {'bleu': [], 'rouge_1': [], 'rouge_2': [], 'rouge_l': [], 
                       'meteor': [], 'bert_score': []},
        'description': {'bleu': [], 'rouge_1': [], 'rouge_2': [], 'rouge_l': [], 
                       'meteor': [], 'bert_score': []}
    }
    
    for task in ['requirement', 'description']:
        print(f"\nProcessing random pairs for {task}")
        task_df = df[df['task'] == task]
        
        if len(task_df) < 2:
            continue
            
        # 무작위 쌍 생성: result와 groundtruth를 섞어서 비교
        all_indices = list(range(len(task_df)))
        pairs = []
        if not num_random_pairs:
            num_random_pairs = len(df)
        for _ in range(num_random_pairs):
            idx1, idx2 = np.random.choice(all_indices, size=2, replace=False)
            pairs.append((idx1, idx2))
            
        # 배치 처리
        for i in trange(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + min(batch_size, len(pairs) - i)]
            
            # 무작위로 섞인 result와 groundtruth 쌍
            predictions = [task_df.iloc[pair[0]]['result'] for pair in batch_pairs]
            references = [task_df.iloc[pair[1]]['groundtruth'] for pair in batch_pairs]
            
            batch_scores = evaluator.evaluate_batch(predictions, references)
            
            for metric_scores in batch_scores:
                random_scores[task]['bleu'].append(metric_scores.bleu)
                random_scores[task]['rouge_1'].append(metric_scores.rouge_1)
                random_scores[task]['rouge_2'].append(metric_scores.rouge_2)
                random_scores[task]['rouge_l'].append(metric_scores.rouge_l)
                random_scores[task]['meteor'].append(metric_scores.meteor)
                random_scores[task]['bert_score'].append(metric_scores.bert_score)
    
    # Print results
    for task in ['requirement', 'description']:
        if random_scores[task]['bleu']:
            print(f"\nRandom Baseline for {task}:")
            for metric, scores in random_scores[task].items():
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print(f"{metric}: {mean_score:.4f} ± {std_score:.4f}")
    
    return random_scores


def process_file_random_diversity(file_path: str, batch_size: int = 32, num_random_pairs: int = 10000):
    """Diversity (Soundedness)에 대한 random baseline"""
    print("Computing random baseline for diversity...")
    evaluator = TextEvaluator(batch_size=batch_size)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    df = pd.read_json(file_path, lines=True)
    
    random_scores = {
        'bleu': [], 'rouge_1': [], 'rouge_2': [], 'rouge_l': [], 
        'meteor': [], 'bert_score': [], 'diversity': []
    }
    
    # 전체 데이터셋에서 무작위 쌍 생성
    all_indices = list(range(len(df)))
    pairs = []
    if not num_random_pairs:
        num_random_pairs = len(df)
    for _ in range(num_random_pairs):
        idx1, idx2 = np.random.choice(all_indices, size=2, replace=False)
        pairs.append((idx1, idx2))
    
    # 배치 처리
    for i in trange(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + min(batch_size, len(pairs) - i)]
        
        # 무작위로 섞인 결과와 groundtruth 쌍
        predictions = [df.iloc[pair[0]]['result'] for pair in batch_pairs]
        references = [make_gt(df.iloc[pair[1]]['groundtruth']) for pair in batch_pairs]
        
        # Soundness metrics
        batch_scores = evaluator.evaluate_batch(predictions, references)
        
        for metric_scores in batch_scores:
            random_scores['bleu'].append(metric_scores.bleu)
            random_scores['rouge_1'].append(metric_scores.rouge_1)
            random_scores['rouge_2'].append(metric_scores.rouge_2)
            random_scores['rouge_l'].append(metric_scores.rouge_l)
            random_scores['meteor'].append(metric_scores.meteor)
            random_scores['bert_score'].append(metric_scores.bert_score)
        
        # Diversity scores
        for pred in predictions:
            embeddings = model.encode(pred.strip("\n"), convert_to_tensor=True)
            random_scores['diversity'].append(float((embeddings ** 2).mean()))
    
    # Print results
    print("\nRandom Baseline for Diversity/Soundness:")
    for metric, scores in random_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric}: {mean_score:.4f} ± {std_score:.4f}")
    
    return random_scores


def random_truthfulness(start: int = 0, batch_size: int = 32, num_random_pairs: int = None):
    """
    모든 평가 항목에 대한 random baseline 계산
    """
    for result_file in FILENAMES[start:]:
        print(f"\n=== Processing {result_file} ===")
        
        # Truthfulness (Requirements & Description)
        truth_file = f"results/eval_truthfulness/{result_file}"
        truth_scores = process_file_random_truthfulness(truth_file, batch_size, num_random_pairs)
        
        
def random_diversity(start: int = 0, batch_size: int = 32, num_random_pairs: int = None):
    for result_file in FILENAMES[start:]:
        print(f"\n=== Processing {result_file} ===")
        div_file = f"results/eval_diversity/{result_file}"
        div_scores = process_file_random_diversity(div_file, batch_size, num_random_pairs)



def diverse_than_realdata(model_name_or_path: str = 'CohereForAI/aya-expanse-32b'):
    data = open_json('data/diverse_than_realdata.json')
    indexes = [[vi['graph_idx'] for vi in v] for k, v in data.items()]
    indexes = [i for idx in indexes for i in idx]
    print(len(indexes)) # 1000
    
    # baseline
    baseline_outputs = [[vi['retrieved'] for vi in v] for k, v in data.items()]
    baseline_outputs = [o for outputs in baseline_outputs for o in outputs]
    baseline_outputs = [[oi['content']['main']+" "+oi['content'].get('detail','') for oi in o] for o in baseline_outputs]
    print(len(baseline_outputs)) # 1000
    
    # model_inference
    model_result = open_json(f'results/eval_diversity/{model_name_or_path.replace("/","_")}.jsonl')
    model_outputs = [[vi for vi in v['result'].split("\n") if 'Here are' not in vi and len(vi) > 5] for v in model_result if v['metadata']['idx'] in indexes]
    print(len(model_outputs)) # 1000
    
    diversity = Diversity('sentence-transformers/all-MiniLM-L6-v2')
    baseline_scores = diversity.evaluate(baseline_outputs)
    model_scores = diversity.evaluate(model_outputs)
    # most_diverse_model_outputs = diversity.the_most_diverse(model_outputs)


if __name__ == '__main__':
    fire.Fire({
        'truthfulness': random_truthfulness,
        'diversity': random_diversity,
        'diverse_than_realdata' : diverse_than_realdata
    })