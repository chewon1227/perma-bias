from mylmeval import open_json, save_json
import fire
from careerpathway.scoring import Similarity, Diversity, delete_prefix, get_gt_from_id
import matplotlib.pyplot as plt
import random
import os

filepaths = {
    'diversity' : [
        'results/eval_diversity/Qwen_Qwen2.5-0.5B-Instruct.jsonl',
        'results/eval_diversity/CohereForAI_aya-expanse-8b.jsonl'
    ],
    'soundedness' : [
        'results/eval_diversity/Qwen_Qwen2.5-0.5B-Instruct.jsonl',
        'results/eval_diversity/gpt-4o.jsonl',
    ],
    'truthfulness' : [ # requirements
        'results/eval_truthfulness/Qwen_Qwen2.5-0.5B-Instruct.jsonl',
        'results/eval_truthfulness/gpt-4o-mini.jsonl',
        
    ],
}


def run(measure: str = 'diversity'):
    saved = []
    score_diffs = []
    low_filepath, high_filepath = tuple(filepaths[measure])
    low_data = open_json(low_filepath)
    high_data = open_json(high_filepath)
    
    if measure == 'diversity':
        metric = Diversity()
        low_results = [[delete_prefix(k) for k in r['result'].split("\n")] for r in low_data]
        high_results = [[delete_prefix(k) for k in r['result'].split("\n")] for r in high_data]
        
        low_scores = metric.evaluate(low_results, return_all=True)
        high_scores = metric.evaluate(high_results, return_all=True)
        random_idxes = random.sample(range(len(low_scores['input_sets'])), 100)
        
        for idx in random_idxes:
            saved.append(
                {
                    'option1' : low_scores['input_sets'][idx],
                    'option2' : high_scores['input_sets'][idx],
                    'scores' : [low_scores['mean_distance'][idx], high_scores['mean_distance'][idx]],
                    'winner' : 'option1' if low_scores['mean_distance'][idx] < high_scores['mean_distance'][idx] else 'option2',
                    'type' : 'diversity'
                }
            )
            score_diffs.append(low_scores['mean_distance'][idx] - high_scores['mean_distance'][idx])
        
    elif measure == 'soundedness':
        metric = Similarity()
        low_results = [[delete_prefix(k) for k in r['result'].split("\n")] for r in low_data]
        high_results = [[delete_prefix(k) for k in r['result'].split("\n")] for r in high_data]
        references = [get_gt_from_id(r["meta"]["graph_id"], open_json('data/evalset/diversity.jsonl')) for r in low_data]
        
        low_scores = metric.group_evaluate(low_results, references, the_most_similar=5)
        high_scores = metric.group_evaluate(high_results, references, the_most_similar=5)
        random_idxes = random.sample(range(len(low_scores['input_sets'])), 10)
        
        for idx in random_idxes:
            saved.append(
                {
                    'option1' : low_scores['input_sets'][idx],
                    'option2' : high_scores['input_sets'][idx],
                    'scores' : [int(low_scores['bert_score'][idx]), int(high_scores['bert_score'][idx])],
                    'winner' : 'option1' if low_scores['bert_score'][idx] > high_scores['bert_score'][idx] else 'option2',
                    'type' : 'soundedness'
                }
            )
        
    elif measure == 'truthfulness':
        metric = Similarity()
        low_results = [r['result'] for r in low_data if r['metadata']['task'] == 'requirement']
        high_results = [r['result'] for r in high_data if r['metadata']['task'] == 'requirement']
        references = [r['groundtruth'] for r in low_data if r['metadata']['task'] == 'requirement']
        
        random_idxes = random.sample(range(len(low_results)), 10)
        low_scores = metric.evaluate(low_results, references, return_all=True)
        high_scores = metric.evaluate(high_results, references, return_all=True)
        
        for idx in random_idxes:
            saved.append(
                {
                    'option1' : low_results[idx],
                    'option2' : high_results[idx],
                    'scores' : [int(low_scores['bert_score'][idx]), int(high_scores['bert_score'][idx])],
                    'winner' : 'option1' if low_scores['bert_score'][idx] > high_scores['bert_score'][idx] else 'option2',
                    'type' : 'truthfulness'
                }
            )
    
    plt.hist(score_diffs, bins=5)
    os.makedirs('data/humaneval', exist_ok=True)
    save_json(saved, f'data/humaneval/{measure}.jsonl')


if __name__ == '__main__':
    fire.Fire(run)