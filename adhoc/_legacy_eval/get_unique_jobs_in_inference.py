import os
from mylmeval.utils import open_json
from collections import Counter
import fire
from typing import List
import random

total_job = list()
abstraction_dict = open_json('/home/iyy1112/workspace/Career-Pathway/data/simplified_job_dict.json')

FILES = [
    # 'Qwen_Qwen2.5-72B-Instruct.jsonl',
    'Qwen_Qwen2.5-32B-Instruct.jsonl',
    'Qwen_Qwen2.5-14B-Instruct.jsonl',
    'Qwen_Qwen2.5-7B-Instruct.jsonl',
    'Qwen_Qwen2.5-3B-Instruct.jsonl',
    'Qwen_Qwen2.5-1.5B-Instruct.jsonl',
    'Qwen_Qwen2.5-0.5B-Instruct.jsonl',
    'CohereForAI_aya-expanse-32b.jsonl',
    'CohereForAI_aya-expanse-8b.jsonl',
    'gpt-4o-mini.jsonl',
    'gpt-4o.jsonl'
]

def unique_jobs_arrow(output):
    non_words = [
        'allow for the ',
        'here',
        'career advancement',
        'Based on the ',
        'important to',
        '10 career steps',
        'best career steps',
        'depends on ',
        'career paths',
        'asking for career progression',
        'assuming you',
        'examples include',
        'important to',
        'best career path',
        'these roles',
        'specific interests',
        'let me know',
        'depending on',
        'they offer'
        ]
    total_job = []
    for job in output:
        # 기본 텍스트 정제
        job = job.split("→")[-1].strip()
        if '. ' in job:
            job = job.split('. ')[1]
        for i in range(1, 11):
            job = job.replace(f"Job {i}", '')
        job = job.replace("*", '').strip()
        job = job.replace("  ", '')
        
        # 조기 필터링 체크
        if 'here' in job.lower():
                continue
            
        if job.endswith('.') or job.endswith('?') or job.endswith(':'):
            continue
            
        if '(' in job:
            job = job.split('(')[0].strip()
            
        # non_words 체크
        if any(word in job.lower() for word in non_words):
            continue
            
        total_job.append(job)
    return total_job 
                         
def unique_jobs(output):
    non_words = [
        'allow for the ',
        'here',
        'career advancement',
        'Based on the ',
        'important to',
        '10 career steps',
        'best career steps',
        'depends on ',
        'career paths',
        'asking for career progression',
        'assuming you',
        'examples include',
        'important to',
        'best career path',
        'these roles',
        'specific interests',
        'let me know',
        'depending on',
        'they offer'
        ]
    total_job = []
    for job in output:
        # 기본 텍스트 정제
        if '. ' in job:
            job = job.split('. ')[1]
        for i in range(1, 11):
            job = job.replace(f"Job {i}", '')
        job = job.replace("*", '').strip()
        job = job.replace("  ", '')
        
        # 조기 필터링 체크
        if 'here' in job.lower():
                continue
            
        if job.endswith('.') or job.endswith('?') or job.endswith(':'):
            continue
            
        if '(' in job:
            job = job.split('(')[0].strip()
            
        # non_words 체크
        if any(word in job.lower() for word in non_words):
            continue
            
        total_job.append(job)
    return total_job


def unique_jobs_abstraction(output):
    non_words = [
        'allow for the ',
        'here',
        'career advancement',
        'Based on the ',
        'important to',
        '10 career steps',
        'best career steps',
        'depends on ',
        'career paths',
        'asking for career progression',
        'assuming you',
        'examples include',
        'important to',
        'best career path',
        'these roles',
        'specific interests',
        'let me know',
        'depending on',
        'they offer'
        ]
    total_job = []
    for job in output:
        # 기본 텍스트 정제
        if '. ' in job:
            job = job.split('. ')[1]
        for i in range(1, 11):
            job = job.replace(f"Job {i}", '')
        job = job.replace("*", '').strip()
        job = job.replace("  ", '')
        
        # 조기 필터링 체크
        if 'here' in job.lower():
                continue
            
        if job.endswith('.') or job.endswith('?') or job.endswith(':'):
            continue
            
        if '(' in job:
            job = job.split('(')[0].strip()
            
        # non_words 체크
        if any(word in job.lower() for word in non_words):
            continue
        
        if job in abstraction_dict:
            total_job.append(abstraction_dict[job])
    return total_job
    
def run(
    result_dir: str = 'results/eval_diversity_1/{}',
    do_write: bool = False,
    do_print: bool = False,
    per_file: bool = False,
    top_n: int = 10,
    ):
    
    if '{}' not in result_dir:
        result_dir = result_dir+'/{}'
        
    def parse_func(output: str, result_dir) -> List:
        if 'gen' in result_dir:
            max_depth = result_dir.count('gen')
            max_depth = int(result_dir.split("/")[-2].split("tmp_")[-1])+1 if 'tmp' in result_dir else max_depth
            leaf_nodes = [r['content'] for r in output['nodes'] if len(r['parent_id']) == max_depth]
            parsing_output = leaf_nodes
            parsing_output = unique_jobs(parsing_output)
        else:
            parsing_output = output['result'].split('\n')
            parsing_output = unique_jobs_arrow(parsing_output)
        return parsing_output
    
    outputs = []
    
    infinigram = open_json('data/data5_jobdict_infinigram_count.json')
    infinigram = [r['en'] for r in infinigram] # dict : jobname, count
    
    file_names = [result_dir.format(r) for r in FILES if os.path.exists(result_dir.format(r))]
    if len(file_names) == 0:
        raise FileNotFoundError(f"No files in {result_dir}")
    for file_name in file_names:
        try:
            data = open_json(file_name)
            outputs.extend(data)
        except:
            continue

    total_job = []
    for output in outputs:
        total_job.extend(parse_func(output, result_dir))
    print('Total jobs:', len(set(total_job)), f"{len(set(total_job))/len(total_job):.4f}")
    for common_job, cnt in Counter(total_job).most_common(top_n):
        print(common_job, cnt, [r for r in infinigram if common_job==r['jobname']])
    
    if do_write:
        open(f'{result_dir}/total_jobs.txt', 'w').write('\n'.join(set(total_job)))

    if do_print:
        sorted_total_job = sorted([(sen, len(sen)) for sen in total_job], key=lambda x: x[1], reverse=True)
        for i in range(0, 10000, 100):
            print(sorted_total_job[i][0])

    if per_file:
        file_names = [r for r in os.listdir(os.path.dirname(result_dir)) if r in FILES]
        for file_name in file_names:
            total_job = list()
            outputs = open_json(result_dir.format(file_name))
            try:
                for output in outputs:
                    total_job.extend(parse_func(output, result_dir))
                    
                print(file_name)
                print(len(set(total_job)), f"{len(set(total_job))/len(total_job):.4f}")
                for common_job, cnt in Counter(total_job).most_common(top_n):
                    print(common_job, cnt, [r for r in infinigram if common_job==r['jobname']])
    
            except:
                print(file_name, 'error')
                continue
    
    
def meta_run(
    do_write: bool = False,
    do_print: bool = False,
    per_file: bool = False,
    do_abstract: bool = False,
):
    result_dirs = [
        'results/eval_diversity_1/{}',
        # 'results/eval_prompt3_100gen_2gen_2gen_2gen_2gen_1000/{}',
        # 'results/eval_prompt3_100gen_2gen_2gen_2gen_2gen_0/{}',
        
        # 'results/4_gar/{}',
        # 'results/5_ragtree/{}',
        # 'results/eval_diversity_40/{}',
        # 'results/baseline_retrieve/{}'
    ]
    for result_dir in result_dirs: 
    # for result_dir in ['/scratch2/iyy1112/results/mcts_value_model/tmp_3_{}']:
        print(f"Processing {result_dir}")
        run(result_dir, do_write, do_print, per_file, do_abstract)
    

if __name__ == '__main__':
    fire.Fire(meta_run)