evaimport fire
import os
import numpy as np
import torch
from dataclasses import dataclass
import nltk
import re
import torch.nn.functional as F
from typing import List, Dict, Tuple
from careerpathway.scoring import Diversity, Similarity
import pandas as pd

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

FILENAMES = [
    'gpt-4o-mini.jsonl',
    'gpt-4o.jsonl',
]

def process_file_truthfulness(file_path: str, batch_size: int = 32):
    similarity = Similarity(batch_size=batch_size)
    df = pd.read_json(file_path, lines=True)
    df['task'] = df['metadata'].apply(lambda x: x['task'])
    
    scores = {
        'requirement': {'bleu': [], 'rouge_1': [], 'rouge_2': [], 'rouge_l': [], 
                       'meteor': [], 'bert_score': []},
        'description': {'bleu': [], 'rouge_1': [], 'rouge_2': [], 'rouge_l': [], 
                       'meteor': [], 'bert_score': []},
        'salary_': [],
        'salary': {}
    }
    
    # Salary evaluation remains the same
    salary = df[df['task'] == 'salary']
    for _, row in salary.iterrows():
        groundtruth, response, lang = row['groundtruth'], row['result'], row['metadata']['language']
        try:
            response_nums = [int(x) for x in re.findall(r'\d+', response.replace(',', ''))]
            reference_num = int(re.findall(r'\d+', str(groundtruth))[0])
            if response_nums:
                response_val = response_nums[0] if len(response_nums) == 1 else sum(response_nums[:2])/2
                scores['salary_'].append((np.abs((float(response_val) - reference_num)), lang))
        except:
            pass
    
    # Process requirements and descriptions with multiple metrics
    for task in ['requirement', 'description']:
        task_df = df[df['task'] == task]
        task_scores = similarity.evaluate(
            task_df['result'].tolist(),
            task_df['groundtruth'].tolist()
        )
        
        scores[task] = task_scores
        
    for task in ['requirement', 'description']:
        print(f"\nMean scores for {task}:")
        for k, v in scores[task].items():
            print(f"{k}: {v}")
    
    if scores['salary_']:
        for lang in set([s[1] for s in scores['salary_']]):
            salary_mean = np.mean([s[0] for s in scores['salary_'] if s[1] == lang])
            scores['salary'][lang] = salary_mean
            print(f"\nMean Salary Score ({lang}): {salary_mean:.4f}")
    # scores.remove('salary_')
    
    return scores


def process_file_truthfulness_lang(file_path: str, batch_size: int = 32):
    print("Processing per language...")
    similarity = Similarity(batch_size=batch_size)
    df = pd.read_json(file_path, lines=True)
    df['task'] = df['metadata'].apply(lambda x: x['task'])
    df['language'] = df['metadata'].apply(lambda x: x['language'])
    
    scores = {
        'requirement': {},
        'description': {},
        'salary': {}
    }
    
    languages = df['language'].unique()
    
    for lang in languages:
        print(f"\nProcessing language: {lang}")
        df_lang = df[df['language'] == lang]
        
        # Initialize scores for this language
        for task in ['requirement', 'description']:
            scores[task][lang] = {
                'bleu': [], 'rouge_1': [], 'rouge_2': [], 'rouge_l': [], 
                'meteor': [], 'bert_score': []
            }
        scores['salary'][lang] = []
        
        # Process salary
        salary = df_lang[df_lang['task'] == 'salary']
        for _, row in salary.iterrows():
            groundtruth, response = row['groundtruth'], row['result']
            try:
                response_nums = [int(x) for x in re.findall(r'\d+', response.replace(',', ''))]
                reference_num = int(re.findall(r'\d+', str(groundtruth))[0])
                if response_nums:
                    response_val = response_nums[0] if len(response_nums) == 1 else sum(response_nums[:2])/2
                    scores['salary'][lang].append(np.abs(float(response_val) - reference_num))
            except:
                pass
        
        # Process requirements and descriptions with multiple metrics
        for task in ['requirement', 'description']:
            task_df = df_lang[df_lang['task'] == task]
            if not task_df.empty:
                task_scores = similarity.evaluate(
                    task_df['result'].tolist(),
                    task_df['groundtruth'].tolist()
                )
                
                # Update scores dictionary with the results
                for metric, values in task_scores.items():
                    scores[task][lang][metric] = values
        
        # Print scores for this language
        for task in ['requirement', 'description']:
            if scores[task][lang]['bleu']:  # Check if we have scores
                print(f"\nMean scores for {task} ({lang}):")
                for metric in scores[task][lang].keys():
                    mean_score = np.mean(scores[task][lang][metric])
                    print(f"{metric}: {mean_score:.4f}")
        
        if scores['salary'][lang]:
            salary_mean = np.mean(scores['salary'][lang])
            print(f"Mean Salary Score ({lang}): {salary_mean:.4f}")
    
    return scores


def truthfulness(
    start: int = 0,
    per_lang: bool = False,
    batch_size: int = 32
    ):
    for result_file in FILENAMES[start:]:
        print(f"\nProcessing {result_file}")
        file_path = f"results/eval_truthfulness/{result_file}"
        if per_lang:
            scores = process_file_truthfulness_lang(file_path, batch_size)
        else:
            scores = process_file_truthfulness(file_path, batch_size)



if __name__ == '__main__':
    fire.Fire(truthfulness)
    
