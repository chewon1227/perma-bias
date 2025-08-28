from careerpathway.scoring import Diversity, Similarity, load_diversity
import fire
from typing import List, Dict, Any
import os
from mylmeval import open_json
import random

def measure_diversity(diversity: Any, evaluate_set: List[List]):
    return diversity.evaluate(evaluate_set)


def measure_similarity(similarity: Any, evaluate_set: List[List], groundtruths: List[List]):
    return similarity.group_evaluate(evaluate_set, groundtruths)


def only_leaf_nodes_tree(data: List[List[Dict]], leaf_n: int | None = None, seed: int = 42):
    result = [[{**di, 'parent_len' : len(di['parent_id'])} for di in d] for d in data]
    # filter out only parent_len is most
    result = [[d for d in di if d['parent_len'] == max([d['parent_len'] for d in di])] for di in result]
    random.seed(seed)
    if leaf_n:
        result = [d for d in result if len(d) >= leaf_n]
        result = [random.sample(d, leaf_n) for d in result]
    result = [[d['content'] for d in di] for di in result]
    return result


def only_leaf_nodes(data: List[List[Dict]], leaf_n: int | None = None):
    total_results = []
    for result in data:
        answer = result['result']
        leaf_nodes = [d.split('→')[-1] for d in answer.split('\n') if '→' in d]
        total_results.append(leaf_nodes)
    return total_results
    
    
def branch_nodes(data: List[List[Dict]], branch_n: int | None = None):
    total_results = []
    for result in data:
        results = []
        max_parent_len = max([len(d['parent_id']) for d in result])
        for node in result:
            if max_parent_len == len(node['parent_id']):
                job = node['content']
                for parent in node['parent_id']:
                    parent = [r for r in result if r['node_id'] == parent]
                    job = parent[0]['content'] + ' -> ' + job
                results.append(job)
        total_results.append(results)
        
            
def run(
    file_dir: str = 'results/diversity/eval_diversity_10gen_2gen_2gen',
    method: str = 'leaf',# branch
    leaf_n: int | None = 10,
    seed: int = 2,
    measure: str = 'diversity' # soundedness
    ):
    if measure == 'diversity':
        diversity = Diversity()
    elif measure == 'soundedness':
        similarity = Similarity()
        testset, _ = load_diversity(test_size=50)
        groundtruths = [' '.join(r['nodes']) for r in testset]
        
    files = os.listdir(file_dir)
    to_process = []
    for file in files:
        if file.endswith('.json') or file.endswith('.jsonl'):
            to_process.append(os.path.join(file_dir, file))
        elif 'tmp' in file and os.path.isdir(os.path.join(file_dir, file)):
            to_process.extend([os.path.join(file_dir, file, f) for f in os.listdir(os.path.join(file_dir, file))])
    for file in to_process:
        data = open_json(file)
        if 'nodes' in data[0]:
            nodes = [r['nodes'] for r in data]
            if method == 'leaf':
                evaluate_set = only_leaf_nodes_tree(nodes, leaf_n=leaf_n, seed=seed)
            elif method == 'branch':
                evaluate_set = branch_nodes(nodes, branch_n=None)
            else:
                raise ValueError('method should be either leaf or branch')
        else:
            if method == 'leaf':
                evaluate_set = only_leaf_nodes(data, leaf_n=leaf_n)
            elif method == 'branch':
                evaluate_set = branch_nodes(data, branch_n=None)
            else:
                raise ValueError('method should be either leaf or branch')
        print(f'Processing {file}')
        if measure == 'diversity':
            measure_diversity(diversity, evaluate_set)
        elif measure == 'soundedness':
            try:
                score = measure_similarity(similarity, evaluate_set, groundtruths)
                print(score)
            except Exception as e:
                print(f"Error processing {file}: {e}")
                print(len(evaluate_set), len(groundtruths))
        else:
            raise ValueError('measure should be either diversity or soundedness')


if __name__ == '__main__':
    fire.Fire(run)