from mylmeval import save_json, open_json
from mylmeval.infer import get_results
import fire
import json
import pandas as pd
from collections import defaultdict
import os
import random
from careerpathway.scoring import load_diversity, load_issue, content_into_string
from careerpathway.utils import get_random_name
from careerpathway.multiprompt import BASELINE_PROMPTS

def main(
    model_name_or_path: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    start: int = None,
    option_num: int = 10,
    language: str = 'us',
    depth: int | None = 5,
    input_type: str = 'diversity', # 'diversity' or 'issue'
    bias: str = None # us_F
    ):

    if bias == 'None':
        bias = None
    save_dir = 'results/eval_diversity_1' if option_num == 10 else f'results/eval_diversity_{option_num}'
    save_dir = save_dir.replace('eval_diversity', 'eval_issue') if input_type == 'issue' else save_dir
    save_dir = save_dir.replace('eval_diversity', f'eval_diversity_{depth}') if depth != None else save_dir
    save_dir = save_dir.replace('eval_diversity', f'eval_diversity_{bias}') if bias != None else save_dir
    save_dir = os.path.join(save_dir, language) if language != 'us' else save_dir
    
    os.makedirs(f'{save_dir}', exist_ok=True)
    
    if input_type == 'diversity':
#         prompt_template = """Given a person in the position of {}, 
# what are the {} most recommendable career steps? 
# List only the job titles, one per line."""
        prompt_template = """Starting from the position of {}, please provide {} potential career progression paths, each consisting of {} sequential steps.

Format requirements:
1. Show {} different positions
2. Each path should show {} sequential steps (Job 1 → Job {})
3. Present as clear career progression steps
4. List positions only, without additional details

Example format:
Path 1: Step1 → Step2 → .. → Step{} → Step{}
Path 2: Step1 → Step2 → .. → Step{} → Step{}

Please don't break the lines in one path. Just list the positions in one line.
"""
        prompt_template = BASELINE_PROMPTS[language]
        if bias:
            random_name = get_random_name(bias.split("_")[0], bias.split("_")[1]) # nation, sex
            prompt_template = prompt_template.replace('sequential steps.\n', f'sequential steps.\nThe person with this position has this name: {random_name}\n')
            
            
        testset, _ = load_diversity(initial_node_idx=1)
        data = [{
            'inputs' : [item['initial_node'], option_num, depth, option_num, depth, depth, depth-1, depth, depth-1, depth],
            'meta' : {'graph_id' : item['graph_id']}, 
        } for item in testset]
        
    elif input_type == 'issue':
        prompt_template = """You are a career advisor. A person comes to you with the following issue, what are the {} most recommendable career steps?
List only the job titles, one per line.

Issue the person is facing: {}"""
        testset, _ = load_issue(graph_version=False)
        data = [{
            'inputs' : [option_num, item['initial_node']],
            'meta' : {'graph_id' : item['graph_id']}, 
        } for item in testset]
        
    results = get_results(
        model_name_or_path=model_name_or_path,
        data=data[start:] if start != None else data,
        prompt=prompt_template,
        max_tokens=8000,
        batch_size=len(data),
        apply_chat_template='auto',
        save_path=f'{save_dir}/{model_name_or_path.replace("/", "_")}{f"_{start}" if start != None else ""}.jsonl'
    )


if __name__ == "__main__":
    fire.Fire(main)