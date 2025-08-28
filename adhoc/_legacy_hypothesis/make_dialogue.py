import fire
from mylmeval import open_json, get_results, MyLLMEval
import os
import numpy as np
from careerpathway.utils import extract_num
import json

def generate_tree_paths(total_nodes=151, levels=[1, 10, 20, 40, 80]):
    """
    주어진 레벨별 노드 수에 따라 트리 구조의 패스를 생성합니다.
    
    Args:
        total_nodes (int): 총 노드의 수
        levels (list): 각 레벨별 노드의 수
    
    Returns:
        list: 각 패스를 리스트로 담은 리스트
    """
    def get_node_ranges():
        # 각 레벨별 노드 범위 계산
        ranges = []
        start = 0
        for level_size in levels:
            end = start + level_size
            ranges.append((start, end))
            start = end
        return ranges
    
    def get_child_indices(parent_idx, parent_level_idx, ranges):
        # 부모 노드의 자식 노드들의 인덱스를 계산
        if parent_level_idx >= len(ranges) - 1:
            return []
            
        parent_position = parent_idx - ranges[parent_level_idx][0]
        next_level_start = ranges[parent_level_idx + 1][0]
        nodes_per_parent = levels[parent_level_idx + 1] // levels[parent_level_idx]
        
        start_idx = next_level_start + (parent_position * nodes_per_parent)
        end_idx = start_idx + nodes_per_parent
        
        return list(range(start_idx, end_idx))
    
    def generate_path(start_idx, ranges):
        # 한 패스의 노드들을 생성
        path = [start_idx]
        current_level = 0
        
        while current_level < len(ranges) - 1:
            children = get_child_indices(path[-1], current_level, ranges)
            if not children:
                break
            path.append(children[0])
            current_level += 1
            
        return path
    
    # 모든 패스 생성
    ranges = get_node_ranges()
    all_paths = []
    
    # 마지막 레벨의 각 노드에 대해 루트까지의 패스를 생성
    last_level_start, last_level_end = ranges[-1]
    for end_node in range(last_level_start, last_level_end):
        path = []
        current_node = end_node
        
        # 각 레벨에서 부모 노드를 찾아 패스를 구성
        for level_idx in range(len(levels) - 1, -1, -1):
            path.insert(0, current_node)
            if level_idx > 0:
                parent_level_size = levels[level_idx - 1]
                nodes_per_parent = levels[level_idx] // parent_level_size
                relative_pos = (current_node - ranges[level_idx][0]) // nodes_per_parent
                current_node = ranges[level_idx - 1][0] + relative_pos
        
        all_paths.append(path)
    
    return all_paths


def main(
    file_path: str = 'results/eval_prompt3_10gen_2gen_2gen_2gen_0/Qwen_Qwen2.5-3B-Instruct_with_requirements.jsonl',
    ):
    
    prompt = """You are a career counselor. The following question has been received, and in your mind, you've imagined the following career paths. Based on these, generate a response.
- The response must include career paths and expected salary ranges
- For each path, list the main requirements needed
- Present the top 3 recommended paths

[User]: {}
[Career Pathways]: {}
"""
    data = open_json(file_path)
    levels = [1, 10, 20, 40, 80]
    total_nodes = 151
    path_ids = generate_tree_paths(total_nodes, levels)
    input_data = []
    for item in data:
        if len(item['nodes']) == 151:
            item_paths = []
            for path_id in path_ids:
                item_paths.append('-'.join([item['nodes'][idx]['content'] for idx in path_id]))
            item['paths'] = '\n'.join([f"Path {i+1}.{r}" for i, r in enumerate(item_paths)])
            input_data.append(item)
            
    results = get_results(
        model_name_or_path='Qwen/Qwen2.5-3B-Instruct',
        prompt = '{}',
        data=[{**d, 'inputs' : [prompt.format(d['initial_node'], d['paths'])]} for d in input_data],
        max_tokens=1024,
        batch_size=len(input_data),
        save_path='tmp.jsonl',
        do_log=True
    )

if __name__ == '__main__':
    fire.Fire(main)