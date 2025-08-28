import fire
from mylmeval import open_json, get_results, MyLLMEval
import os
import numpy as np
from careerpathway.utils import extract_num
import json

def main(
    file_path: str = 'results/eval_prompt3_10gen_2gen_2gen_2gen_0/',
    per_stage: bool = False,
    do_log: bool = False,
    type: str = 'requirements', # requirements or salary
    do_baseline: bool = False,
    model_path_given: str | None = None
    ):
    
    if type == 'salary':
        if model_path_given:
            model_path = model_path_given
            main_func = lambda x: get_salary(x, Myllmeval, do_log, prompt="Answer the salary for the following job.\n{}Biofuels Production Managers: USD 116970\n{}")
        else:
            model_path = '/scratch2/snail0822/career/job-salary-3b-per'
            main_func = lambda x: get_salary(x, Myllmeval, do_log)
        logging_func = lambda x: print(f"Min: {np.mean(x[0])}, Max: {np.mean(x[1])}")
    elif type == 'requirements':
        if model_path_given:
            model_path = model_path_given
            main_func = lambda x: get_requirements(x, Myllmeval, do_log)
        else:
            model_path = '/scratch2/snail0822/career/job-requirements-3b'
            main_func = lambda x: get_requirements(x, Myllmeval, do_log)
        logging_func = lambda x: print(f"Requirements: {x}")
    
        
    Myllmeval = MyLLMEval(model_path = model_path)
    if not do_baseline:
        data = open_json(os.path.join(file_path, 'Qwen_Qwen2.5-3B-Instruct.jsonl'))
    else:
        per_stage = False
        data = open_json('/home/iyy1112/workspace/Career-Pathway/data/evalset/truthfulness.jsonl')
        data = [r for r in data if (r['language'] == 'en') and (r['task'] == type if type == 'salary' else 'requirement')]
        data = [{'inputs': [r['input']], 'groundtruth' : r['groundtruth'], 'stage': 0} for r in data]

    if do_baseline:
        results = main_func(data)
        logging_func(results)
        return
            
    elif 'nodes' in data[0]:
        # Create mapping from node content to results
        results_map = {}
        
        # Prepare input data for inference
        input_data = [[{'inputs': [k['content']], 'stage': len(k['parent_id'])} for k in r['nodes']] for r in data]
        flat_input_data = [r for k in input_data for r in k]
        
        # Get inference results
        results = main_func(flat_input_data)
        
        # Map results back to original data structure
        current_idx = 0
        for item in data:
            for node in item['nodes']:
                if type == 'salary':
                    node['min_salary'] = results[0][current_idx]
                    node['max_salary'] = results[1][current_idx]
                else:
                    node['requirements'] = results[current_idx]
                current_idx += 1
        
        # Save updated data back to file
        output_path = os.path.join(file_path, f'Qwen_Qwen2.5-3B-Instruct_with_{type}.jsonl')
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
                
        # Print summary if needed
        if per_stage:
            for i in range(max(node['stage'] for item in data for node in item['nodes']) + 1):
                stage_results = [r for idx, r in enumerate(results) if flat_input_data[idx]['stage'] == i]
                logging_func(stage_results)
        else:
            logging_func(results)
            
    else:
        # Original processing for non-tree structure data
        input_data = []
        if file_path in ['results/eval_diversity_1']:
            for item in data:
                for k in [p for p in item['result'].split("\n") if len(p) > 2]:
                    for i in range(0, 10):
                        try:
                            input_data.append({'inputs': [k.strip()], 'stage': i})
                        except:
                            pass
        else:
            for item in data:
                for k in [p for p in item['result'].split("\n") if '→' in p]:
                    for i in range(0, 5):
                        try:
                            input_data.append({'inputs': [k.split("→")[i].strip()], 'stage': i})
                        except:
                            pass
                            
        if per_stage:
            for i in range(0, file_path.count('gen')):
                data_stage = [r for r in input_data if r['stage'] == i]
                results = main_func(data_stage)
                logging_func(results)
        else:
            results = main_func(input_data)
            logging_func(results)

def get_requirements(
    data,
    evaluator,
    do_log = False,
    prompt = "Answer the requirements for the following job.\n{}"
):
    results = evaluator.inference(
        data=data,
        prompt=prompt,
        do_log=False,
        max_tokens=512,
        batch_size=len(data),
        save_path='tmp.json',
    )

    if do_log:
        for item, result in zip(data, results):
            print(item['inputs'][0])
            print(result)
        
    return results

def get_salary(
    data,
    evaluator,
    do_log = False,
    prompt = "Answer the salary for the following job.\n{}"
):
    results = evaluator.inference(
        data=data,
        prompt=prompt,
        do_log=False,
        batch_size=len(data),
        save_path='tmp.json',
    )

    if do_log:
        for item, result in zip(data, results):
            print(item['inputs'][0])
            print(result)
        
    min_mean = [extract_num(r.split(" - ")[0]) for r in results]
    max_mean = [extract_num(r.split(" - ")[-1]) for r in results]
    min_mean = [r for r in min_mean if r != None]
    max_mean = [r for r in max_mean if r != None]
    
    if 'groundtruth' in data[0]:
        difference = [abs(extract_num(r['groundtruth']) - (min_mean[i] + max_mean[i]) / 2) for i, r in enumerate(data)]
        print(f"Mean difference: {np.mean(difference)}")
    return min_mean, max_mean

if __name__ == '__main__':
    fire.Fire(main)