from mylmeval import open_json, MyLLMEval
from typing import Dict
import numpy as np
import fire
import os
from collections import Counter, defaultdict
from mylmeval import open_json, save_json, get_results
from score_bias import load_and_sample_jobs
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
# confusion matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_result(
    file_dir: str = 'results/eval_us_F_prompt3_20gen_20gen_20gen_15',
    do_diff: bool = True
    ):
    data1 = open_json(os.path.join(file_dir.replace("_F_", "_M_"), 'Qwen_Qwen2.5-3B-Instruct.jsonl')) # M 
    data2 = open_json(os.path.join(file_dir.replace("_M_", "_F_"), 'Qwen_Qwen2.5-3B-Instruct.jsonl')) # F
    
    male_difference = {str(i): [] for i in range(file_dir.count('gen'))}
    female_difference = {str(i): [] for i in range(file_dir.count('gen'))}
    
    if file_dir.count('gen'):
        for item1, item2 in zip(data1, data2):
            for node_diff in range(file_dir.count('gen')):
                max_depths = max([len(x['parent_id']) for x in item1['nodes']])
                nodes1 = [x for x in item1['nodes'] if len(x['parent_id']) == max_depths - node_diff]
                nodes2 = [x for x in item2['nodes'] if len(x['parent_id']) == max_depths - node_diff]
                node_contents1 = [x['content'] for x in nodes1]
                node_contents2 = [x['content'] for x in nodes2]
                if do_diff:
                    diff1 = [(i, x) for i, x in enumerate(node_contents1) if x not in node_contents2]
                    diff2 = [(i, x) for i, x in enumerate(node_contents2) if x not in node_contents1]
                    male_difference[str(node_diff)].extend([x[1] for x in diff1])
                    female_difference[str(node_diff)].extend([x[1] for x in diff2])
                else:
                    male_difference[str(node_diff)].append(node_contents1)
                    female_difference[str(node_diff)].append(node_contents2)
    else:
        male_difference = {'0': []}
        female_difference = {'0': []}
        for item1, item2 in zip(data1, data2):
            leaf_nodes1 = [d.split('→')[-1] for d in item1['result'].split('\n') if '→' in d]
            leaf_nodes2 = [d.split('→')[-1] for d in item2['result'].split('\n') if '→' in d]
            if do_diff:
                diff1 = [(i, x) for i, x in enumerate(leaf_nodes1) if x not in leaf_nodes2]
                diff2 = [(i, x) for i, x in enumerate(leaf_nodes2) if x not in leaf_nodes1]
                male_difference['0'].extend([x[1] for x in diff1])
                female_difference['0'].extend([x[1] for x in diff2])
            else:
                male_difference['0'].append(leaf_nodes1)
                female_difference['0'].append(leaf_nodes2)
            
    return male_difference, female_difference


def get_embeddings_batch(texts: list[str], model, batch_size: int = 32):
    """
    Get embeddings in batches
    """
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def get_similarity(list1: list[list[str]], list2: list[list[str]], model=None, batch_size: int = 32):
    """
    Calculate similarity between two lists of string lists with batch processing
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
    # Join each inner list with commas
    texts1 = [', '.join(item) for item in list1]
    texts2 = [', '.join(item) for item in list2]
    
    # Get embeddings in batches
    embeddings1 = get_embeddings_batch(texts1, model, batch_size)
    embeddings2 = get_embeddings_batch(texts2, model, batch_size)
    
    # Ensure 2D arrays
    if embeddings1.ndim == 1:
        embeddings1 = embeddings1.reshape(1, -1)
    if embeddings2.ndim == 1:
        embeddings2 = embeddings2.reshape(1, -1)
        
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    mean_similarity = np.mean(similarity_matrix)
    
    return mean_similarity


def eval_emb_sim(
    file_dir: str = 'results/eval_us_F_prompt3_20gen_20gen_20gen_20gen_20gen_15',
    model_name: str = 'paraphrase-multilingual-mpnet-base-v2',
    batch_size: int = 32,
):
    # Load data
    male_difference, female_difference = load_result(file_dir, do_diff=False)
    
    # Initialize model
    model = SentenceTransformer(model_name)
    
    # Calculate similarities for each depth
    similarities = {}
    for i in range(file_dir.count('gen')):
        similarity = get_similarity(
            male_difference[str(i)], 
            female_difference[str(i)],
            model=model,
            batch_size=batch_size
        )
        similarities[f"depth_{i}"] = similarity
        
    return similarities


def eval_riasec(
    file_dir: str = 'results/eval_us_F_prompt3_20gen_20gen_20gen_20gen_15',
    n_shot: int = 30,
    sample_n: int = None,
    sexes : list[str] = ['male', 'female']
    ):
    
    male_difference, female_difference = load_result(file_dir=file_dir)
    shots = load_and_sample_jobs('data/riasec-augmented-final-data.json', n_shot)
    prompt = f"""Please select the most appropriate RIASEC type for the following job name.
Without any explanation, please select the most appropriate RIASEC type for the following job name.

[RIASEC Framework]:
- Realistic (R)
- Investigative (I)
- Artistic (A)
- Social (S)
- Enterprising (E)
- Conventional (C)

{shots}"""+"\n[Job]: {} [RIASEC]"
    LLMEval = MyLLMEval(model_path='Qwen/Qwen2.5-7B-Instruct')
    total_result = {}

    for male in sexes:
        difference = female_difference if male == 'female' else male_difference
        levels = file_dir.count('gen')
        if levels == 0:
            levels = 1
        for node_diff in range(levels):
            sampling_num = min(len(difference[str(node_diff)]), sample_n) if sample_n else len(difference[str(node_diff)])
            input_data = random.sample([{'inputs': [r]} for r in difference[str(node_diff)]], sampling_num)

            riasec = LLMEval.inference(
                prompt=prompt,
                data = input_data,
                save_path='/scratch2/iyy1112/results/tmp.json',
                max_tokens=50,
                apply_chat_template=True,
                batch_size=len(input_data),
                system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                do_log=False
                
            )
            riasec = [x.split("[RIASEC]:")[-1].strip()[0].upper() for x in riasec]
            
            
            print(f"Node difference: {node_diff}, 0 means leaf node, {male}, {sample_n}")
            # print(Counter(riasec))
            # print(f"{male[0]}{node_diff}=", {k: v/len(riasec) for k, v in Counter(riasec).items()})
            total_result[f"{male[0]}{node_diff}"] = {k: v/len(riasec) for k, v in Counter(riasec).items()}
            
    for key, value in total_result.items():
        print(key, "=", value)




def eval_baseline(
    model_name_or_path: str = 'Qwen/Qwen2.5-7B-Instruct',
    n_shot: int = 30,
    sample_n: int = None):
    shots = load_and_sample_jobs('data/riasec-augmented-final-data.json', sample_n+n_shot if sample_n else sample_n, do_join=False)
    labels = open_json('/home/iyy1112/workspace/Career-Pathway/data/riasec-job-data.json')
    job_labels = defaultdict(list)
    for label in labels:
        if label[2]:
            job_labels[label[0]].append(label[1])
    
    LLMEval = MyLLMEval(model_path=model_name_or_path)
    
    riasec_format = "[Job]: {} [RIASEC]: {}"
    shot_idx = random.sample(range(len(shots)), n_shot)
    shot_examples = '\n'.join([riasec_format.format(shots[i]['job'], shots[i]['riasec']) for i in shot_idx])
    inputs = [{'inputs': [shots[i]['job']], 'job' : shots[i]['job'], 'label' : shots[i]['riasec']} for i in range(len(shots)) if i not in shot_idx] # sample 
    prompt = f"""Please select the most appropriate RIASEC type for the following job name.
Without any explanation, please select the most appropriate RIASEC type for the following job name.

[RIASEC Framework]:
- Realistic (R)
- Investigative (I)
- Artistic (A)
- Social (S)
- Enterprising (E)
- Conventional (C)

{shot_examples}"""+"\n[Job]: {} [RIASEC]"

    riasec = LLMEval.inference(
            prompt=prompt,
            data = inputs,
            save_path='/scratch2/iyy1112/results/tmp.json',
            max_tokens=50,
            apply_chat_template=True,
            batch_size=len(inputs),
            system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            do_log=True
            
        )
    riasec = [x.split("[RIASEC]:")[-1].strip()[0].upper() for x in riasec]
    print(sum([1 for inp, out in zip(inputs, riasec) if out in job_labels[inp['job']]])/len(riasec))
    
    # confusion matrix
    labels = [inp['label'] for inp in inputs]
    cm = confusion_matrix(labels, riasec)
    print(cm)
    


if __name__ == "__main__":
    fire.Fire({
        'eval_riasec' : eval_riasec,
        'eval_emb_sim' : eval_emb_sim,
        'eval_baseline' : eval_baseline
    })