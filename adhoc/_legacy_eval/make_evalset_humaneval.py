import fire
import numpy as np
import random
from mylmeval.utils import open_json, save_json
from mylmeval.infer import get_results
from careerpathway.scoring.similarity import Similarity
from careerpathway.scoring.diversity import Diversity
from itertools import combinations
from typing import List, Tuple, Dict
import os

_EXP = {
    'more costful' : 'require more time, money, or effort (e.g., more training, more education, more experience)',
    'more probable' : 'more likely to happen (e.g., more common, more frequent, more expected)',
}


def print_file(save_path, n=60, print_answer=True):
    data = open_json(save_path)
    for i, item in enumerate(data):
        print(f"{i+1}: {item['prompt']}")
        if print_answer:
            print(f"Answer: {item['answer']}")
        print('------------')
        if i >= n:
            break


def transition_cost(
    file_path: str = 'results/transition_cost.jsonl',
    save_path: str = 'data/humaneval/transition_cost.jsonl',    
    keyword: str = 'more costful',
    top_k: int = 50,
    threshold: float = 0.4
):
    os.makedirs('data/humaneval', exist_ok=True)
    # remove
    if os.path.exists(save_path):
        os.remove(save_path)
        
    costs = open_json(file_path)
    print(f"Total {len(costs)} transition costs are loaded from {file_path}")
    combis = list(combinations(costs, 2))
    if top_k:
        random.shuffle(combis)
    prompt = """What kind of career transition would be [{}]?
{}: {}

A. {} → {}
B. {} → {}
"""
    cnt = 0
    q_cnt = 0
    for i, (cost1, cost2) in enumerate(combis):
        source1, target1, emb_cost1 = cost1['source'], cost1['target'], cost1['result']
        source2, target2, emb_cost2 = cost2['source'], cost2['target'], cost2['result']
        # Skip if the difference is smaller than 0.2
        if abs(emb_cost1 - emb_cost2) < threshold or source1[:3] == target1[:3] or source2[:3] == target2[:3] \
            or 'Doctor' in target1 or 'Doctor' in target2 or 'Engineer' in target2 or 'Engineer' in target1:
            q_cnt += 1
            continue
        answer = 'A' if emb_cost1 > emb_cost2 else 'B'
        print(f"{i+1}/{len(combis)}: {source1} → {target1} vs {source2} → {target2}")
        print(f"1: {emb_cost1:.4f}, 2: {emb_cost2:.4f}")
        if top_k and cnt >= top_k:
            break
        save_json([{'prompt' : prompt.format(keyword, keyword.capitalize(), _EXP[keyword], source1, target1, source2, target2), 'answer': answer, 'option1' : cost1, 'option2' : cost2}], save_path, save_additionally=True)
        cnt += 1
        
    print(f"Total {cnt}/{q_cnt} prompts are saved at {save_path}")
    print_file(save_path)
    
    
def transition_probability(
    file_path: str = 'results/transition_probability.jsonl',
    save_path: str = 'data/humaneval/transition_probability.jsonl',
    threshold: float = 0.01
):
    transition_cost(
        file_path = file_path,
        save_path = save_path,
        keyword = 'more probable',
        threshold = threshold
    )
    
    
def bertscore(
    file_paths: List = ['results/eval_truthfulness/Qwen_Qwen2.5-32B-Instruct.jsonl', 'results/eval_truthfulness/gpt-4o.jsonl'],
    save_path: str = 'data/humaneval/bertscore.jsonl', 
    top_k: int = 50,
    threshold: float = 0.03,
    do_llm: bool = False,
    ):
    
    if do_llm:
        prompt = """This is a prompt.
It can be multiligual, so translate the following text into English if needed.
- If there is non-English text, please translate the whole sentence of paragraph into English.
- If there is not any non-English text, please write 'No translation needed'.

[Prompt]
{}

[Translation]:"""
    
        data = open_json(save_path)
        translations = get_results(
            model_name_or_path='gpt-4o-mini',
            prompt=prompt,
            data=[{'inputs' : [item['prompt']], **item} for item in data],
            save_path='tmp.json',
            max_tokens=500,
            temperature=0.7
        )
        
        save_json(
            [{**item, 'prompt' : item['prompt'] if 'No translation needed' in tran else item['prompt'] + f'\n\n[Translation]: {tran}'} for item, tran in zip(data, translations)],
            save_path,
            save_additionally=False
        )
        return
        
    
    similarity = Similarity()
    os.makedirs('data/humaneval', exist_ok=True)
    # remove
    if os.path.exists(save_path):
        os.remove(save_path)
        
    prompt = """Which answer is more similar to the reference?

[User]
Give me a job description of [{}].

[Reference]
{}
-------------------
[Answer A]
{}

[Answer B]
{}"""
    data1 = open_json(file_paths[0])
    data2 = open_json(file_paths[1])
    
    # 'job' : item['metadata']['else']['job']
    data1 = [{'pred' : item['result'].split("Description: ")[-1], 'ref' : item['groundtruth'], 'job' : item['prompt'].split("please provide information for the following job:\nJob: ")[-1].split("\n")[0]} for item in data1 if item['metadata']['language'] == 'en' and item['metadata']['task'] == 'description']
    data2 = [{'pred' : item['result'].split("Description: ")[-1], 'ref' : item['groundtruth'], 'job' : item['prompt'].split("please provide information for the following job:\nJob: ")[-1].split("\n")[0]} for item in data2 if item['metadata']['language'] == 'en' and item['metadata']['task'] == 'description']
    print(f"Total {len(data1)} and {len(data2)} data are loaded from {file_paths[0]} and {file_paths[1]}")
    
    similarities_1 = similarity.evaluate([item['pred'] for item in data1], [item['ref'] for item in data1], return_all=True)['bert_score']
    similarities_2 = similarity.evaluate([item['pred'] for item in data2], [item['ref'] for item in data2], return_all=True)['bert_score']
    differences = [sim1 - sim2 for sim1, sim2 in zip(similarities_1, similarities_2)]
    print(len([d for d in differences if abs(d) > threshold]))
    
    sorted_data = sorted(zip(data1, similarities_1, data2, similarities_2, differences), key=lambda x: len(x[0]['pred']))
    cnt = 0
    
    for i, (item1, sim1, item2, sim2, diff) in enumerate(sorted_data):
        if item1['job'] != item2['job']:
            print(item1['job'], item2['job'])
        if abs(diff) < threshold:
            continue
        answer = 'A' if diff > 0 else 'B'
        job = item1['job']
        print(f"{i+1}/{len(data1)}: {sim1:.4f} vs {sim2:.4f}")
        if top_k and cnt >= top_k:
            break
        if random.random() > 0.5:
            item1, item2 = item2, item1
            sim1, sim2 = sim2, sim1
            answer = 'A' if sim1 > sim2 else 'B'
        save_json([{'prompt' : prompt.format(
            job,
            item1['ref'], 
            item1['pred'], 
            item2['pred']
            ), 'answer': answer, 'option1' : item1, 'option2' : item2}], save_path, save_additionally=True)
        cnt += 1

    print(f"Total {cnt} prompts are saved at {save_path}")    
    print_file(save_path, n=10)
    
    
def diversity(
    file_paths: List = ['results/eval_diversity_1/gpt-4o.jsonl', 'results/eval_diversity_1/Qwen_Qwen2.5-3B-Instruct.jsonl'],
    save_path: str = 'data/humaneval/diversity.jsonl', 
    top_k: int = 50,
    threshold: float = 0.15,
    do_llm: bool = False
    ):

    
    if do_llm:
        prompt = """This is a prompt.
It can be multiligual, so translate the following text into English if needed.
- If there is non-English text, please translate the whole sentence of paragraph into English.
- If there is not any non-English text, please write 'No translation needed'.

[Prompt]
{}

[Translation]:"""
    
        data = open_json(save_path)
        translations = get_results(
            model_name_or_path='gpt-4o-mini',
            prompt=prompt,
            data=[{'inputs' : [item['prompt']], **item} for item in data],
            save_path='tmp.json',
            max_tokens=500,
            temperature=0.7
        )
        
        save_json(
            [{**item, 'prompt' : item['prompt'] if 'no translation needed' in tran.lower() else item['prompt'] + f'\n\n[Translation]: {tran}'} for item, tran in zip(data, translations)],            
            save_path,
            save_additionally=False
        )
        return
        
    
    diversity = Diversity()
    os.makedirs('data/humaneval', exist_ok=True)
    # remove
    if os.path.exists(save_path):
        os.remove(save_path)
        
    prompt = """Which group has more 'Semantic Diversity' in the generated text?
It means the options in the group are more different from each other (e.g., more diverse, more varied, more distinct, heterogeneous).

[A Group]
{}


[B Group]
{}"""
    data1 = open_json(file_paths[0])
    data2 = open_json(file_paths[1])
    data1 = [[k.split(". ")[-1].strip() for k in item['result'].split("\n") if len(k)>3] for item in data1]
    data2 = [[k.split(". ")[-1].strip() for k in item['result'].split("\n") if len(k)>3] for item in data2]
    
    diversity1 = diversity.evaluate(data1, return_all=True)['mean_distance']
    diversity2 = diversity.evaluate(data2, return_all=True)['mean_distance']
    differences = [div1 - div2 for div1, div2 in zip(diversity1, diversity2)]
    print(len([d for d in differences if abs(d) > threshold]))
    
    cnt = 0
    for i, (item1, div1, item2, div2) in enumerate(zip(data1, diversity1, data2, diversity2)):
        if abs(div1 - div2) < threshold:
            continue
        answer = 'A' if div1 > div2 else 'B'
        print(f"{i+1}/{len(data1)}: {div1:.4f} vs {div2:.4f}")
        if top_k and cnt >= top_k:
            break
        # randomly exchange the order of the items
        if random.random() > 0.5:
            item1, item2 = item2, item1
            div1, div2 = div2, div1
            answer = 'A' if div1 > div2 else 'B'
        save_json([{'prompt' : prompt.format('\n'.join(item1), '\n'.join(item2)), 'answer': answer, 'option1' : {'options' : item1, 'diversity' : div1} , 'option2' : {'options' : item2, 'diversity' : div2}}], save_path, save_additionally=True)
        cnt += 1
        
    print(f"Total {cnt} prompts are saved at {save_path}")
    print_file(save_path, n=10)
    
    
def connectd3(
    file_paths: List = ['results/eval_diversity_5_1/Qwen_Qwen2.5-32B-Instruct.jsonl', 'results/eval_prompt3_30gen_30gen_30gen_30gen_80/Qwen_Qwen2.5-3B-Instruct.jsonl'],
    save_path: str = 'data/humaneval/connectd3.jsonl',
    do_llm: bool = True,
    model_name_or_path: str = 'gpt-4o-mini',
    tmp_save_path: str = 'tmp_connectd3_.jsonl'
):
    os.makedirs('data/humaneval', exist_ok=True)
    # remove
    if os.path.exists(save_path):
        os.remove(save_path)
        
    user_prompt = """Please indentify which one is better.
[Metrics]
1. Diversity: which one offers more diverse jobs or career paths?
2. Soundness: which one is more realistic and grounded in reality?
3. Helpfulness: which one is more helpful for the user? Overall assessment for the career recommendation. (e.g., more informative, more insightful, more useful)
    
[Option A Answer]
{}

[Option B Answer] 
{}"""
    llm_process_prompt = """This is a career trajectory anticipation result by LLM. Within the information, please consolidate into a career recommendation paragraph.
- Don't make a new things, just summarize the information, and pinpoint the career recommendation.
- Please make it concise and clear.
    
[User input]:
{}

[LLM anticipation]:
{}

[Recommendation]:
"""

    data1 = open_json(file_paths[0])
    data2 = open_json(file_paths[1])
    chosen = []
    for item1 in data1:
        for item2 in data2:
            if item1['meta']['graph_id'] == item2['graph_id']:
                chosen.append({
                    'inputs' : [item2['initial_node'], str(item1['result'])],
                    'graph_id' : item2['graph_id'], 
                    'initial_node' : item2['initial_node'],
                    'type' : 'baseline',
                })
                chosen.append({
                    'inputs' : [item2['initial_node'], str(item2['nodes'])],
                    'graph_id' : item1['meta']['graph_id'], 
                    'initial_node' : item2['initial_node'],
                    'type' : 'ours',
                })
    
    print(len(chosen))
    if do_llm:
        get_results(
            # model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
            model_name_or_path=model_name_or_path,
            prompt=llm_process_prompt,
            data=chosen,
            save_path=tmp_save_path,
            max_tokens=500,
            temperature=0.7
        )
        return
    
    result = open_json(tmp_save_path)
    for i in range(0, len(result), 2):
        item1 = result[i]
        item2 = result[i+1]
        answer = 'B'
        if random.random() > 0.5:
            item1, item2 = item2, item1
            answer = 'A'
        save_json([
            {
                'prompt' : user_prompt.format(item1['result'], item2['result']), 
                'answer' : answer,
                'option1' : item1, 
                'option2' : item2
                }
            ], 
                  save_path, save_additionally=True)

    print(f"Total {len(data1)} prompts are saved at {save_path}")
    print_file(save_path, print_answer=False)


def save_as_json():
    files = [
        'data/humaneval/transition_cost.jsonl',
        'data/humaneval/transition_probability.jsonl',
        'data/humaneval/diversity.jsonl',
        'data/humaneval/bertscore.jsonl',
        'data/humaneval/connectd3.jsonl'
    ]
    for file in files:
        data = open_json(file)
        save_json(data, file.replace('.jsonl', '.json'))
        print(f"{file} is saved as {file.replace('.jsonl', '.json')}")
        

if __name__ == '__main__':
    fire.Fire({
        'transition_cost': transition_cost,
        'transition_probability' : transition_probability,
        'bertscore' : bertscore,
        'diversity' : diversity,
        'connectd3' : connectd3,
        'save_as_json' : save_as_json
    })
    save_as_json()