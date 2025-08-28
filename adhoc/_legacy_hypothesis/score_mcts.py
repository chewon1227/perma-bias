from careerpathway.scoring import Diversity, Similarity, FILENAMES, get_gt_from_id
from dataclasses import dataclass
from tqdm import tqdm
import random
import os
import fire
from mylmeval import open_json
from typing import Iterator, Dict, Any, Union, List, Tuple
from tqdm import tqdm
from dataclasses import dataclass

def tree_to_path(nodes: List[Dict], root_id: int = 0) -> List[Tuple[str, Dict[str, float]]]:
    paths: List[Tuple[str, Dict[str, float]]] = []
    
    def dfs(current_id: int, current_path: List[str], cumulative_values: Dict[str, float]):
        current_node = nodes[current_id]
        
        # 현재 노드의 position 추가
        position = current_node['position'] if ': ' not in current_node['position'] else current_node['position'].split(': ')[1]
        current_path.append(position)
        
        # 현재 노드의 values를 누적
        current_values = current_node['values']
        for key in current_values:
            cumulative_values[key] = cumulative_values.get(key, 0) + current_values[key] if current_values[key] else 0
        
        if not current_node['children_idx']:  # Leaf node
            paths.append((" -> ".join(current_path), dict(cumulative_values)))
            
        for child_id in current_node['children_idx']:
            # 각 자식 노드에 대해 현재까지의 누적값 복사하여 전달
            dfs(child_id, current_path.copy(), dict(cumulative_values))
    
    dfs(root_id, [], {})
    return paths


def nodes_into_tree_paths(nodes_data: List[Dict[str, Any]], type: int = 1) -> List[str]:
    if type == 1:  # sampled 10 / 80
        paths = tree_to_path(nodes_data)
        paths = random.sample(paths, min(10, len(paths)))  # 10개
        paths = [r[0] for r in paths]  # 경로 문자열만 반환
    elif type == 2:  # best path
        paths = tree_to_path(nodes_data)
        # 누적된 values의 expected_salary를 기준으로 정렬
        paths = sorted(paths, key=lambda x: x[1]['expected_salary'], reverse=True)
        paths = [r[0] for r in paths]
    return paths
        

def process_file(file_path: str, random_num: int = None, type: int = 1) -> tuple[List[List[str]], List[str]]:
    # Load diversity file and main data
    diversity_file = open_json('data/evalset/diversity.jsonl')
    data = open_json(file_path)
    print(f"Loaded {len(data)} samples")
    
    # Sample random entries if specified
    if random_num:
        data = random.sample(data, random_num)
    
    predictions = []
    references = []
    
    for sample in tqdm(data):
        references.append(get_gt_from_id(sample['graph_id'], diversity_file))
        predictions.append(nodes_into_tree_paths(sample['nodes'], type=type))
        
    return predictions, references



def score_diversity_and_soundedness(
    file_name_format: str = '/scratch2/iyy1112/results/mcts_value_model/tmp_1_{}_1',
    type: int = 1
):
    diversity = Diversity('sentence-transformers/all-MiniLM-L6-v2')
    for result_file in FILENAMES:
        file_path = file_name_format.format(result_file)
        if os.path.exists(file_path):
            print(f"\nProcessing {result_file}")
    
        if type == 1.5:
            total_paths = []
            for id in range(0, len(data), 40):
                paths = []
                results = data[id:id+40]
                for result in results:
                    paths.extend([
                        r for r in result['result'].split("\n") if len(r) > 5
                    ])
                    
                total_paths.append(random.sample(paths, 10))
                
            diversity_score = diversity.evaluate(total_paths)
            print(f"Diversity: {diversity_score}")
            return diversity_score
        
        else:
            predictions, references = process_file(file_path, type=type)
            diversity_score = diversity.evaluate(predictions)
            print(f"Diversity: {diversity_score}")


            
            
            
if __name__ == '__main__':
    fire.Fire(score_diversity_and_soundedness)