import graphviz
from mylmeval import open_json
from careerpathway.scoring import Diversity, Similarity, get_gt_from_id, delete_prefix
import fire
import random
from typing import List, Dict
import multiprocessing as mp
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet


diversity_file = open_json('data/evalset/diversity.jsonl')

FILENAMES = [
    # 'Qwen_Qwen2.5-0.5B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-1.5B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-3B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-7B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-14B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-32B-Instruct.jsonl',
    'CohereForAI_aya-expanse-8b.jsonl',
    'CohereForAI_aya-expanse-32b.jsonl',
    # 'Qwen_Qwen2.5-72B-Instruct.jsonl',
]


def visualize_career_path(data, output_path="career_tree", return_paths=False, return_paths_type=0):
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Career Path')
    dot.attr(rankdir='TB')  # Top to Bottom layout

    # Function to create node label
    def create_label(node):
        position = node['position']
        if ': ' in position:
            position = position.split(': ')[1]
        if 'values' in node and 'expected_salary' in node['values']:
            return f"{position}\nExp: {node['years_experience']}yr, {node['node_id']}\nSalary: ${node['values']['expected_salary']/1000:.1f}k\nFitness: {node['values']['career_fit']:.2f}"
        else:
            return f"{position}\nExp: {node['years_experience']}yr, {node['node_id']}"

    # Add nodes with different colors for leaf nodes
    for node in data['nodes']:
        node_id = node['node_id']
        is_leaf = not node['children_idx']
        fillcolor = 'lightblue' if is_leaf else 'white'
        if is_leaf and 'values' in node:
            node['values']['accumulative_salary'] = sum(
                [float(r['values']['expected_salary']) for r in data['nodes'] if node_id in r['children_idx']]
            ) + node['values']['expected_salary']
        
        # Create node
        dot.node(
            str(node['node_id']),
            create_label(node),
            style='filled',
            fillcolor=fillcolor
        )

    # Add edges
    for node in data['nodes']:
        for child_idx in node['children_idx']:
            dot.edge(str(node['node_id']), str(child_idx))

    if return_paths:
        if return_paths_type == '80': # Total 80
            paths = [r['position'] for r in data['nodes'] if not r['children_idx']]
        elif return_paths_type == '10': # Random 10
            try:
                paths = random.sample([r['position'] for r in data['nodes'] if not r['children_idx']], min(10, len(data['nodes'])))
            except:
                paths = [r['position'] for r in data['nodes'] if not r['children_idx']]
        elif return_paths_type == '40': # Random 40
            paths = random.sample([r['position'] for r in data['nodes'] if not r['children_idx']], min(40, len(data['nodes'])))
        elif return_paths_type == 'best_salary_10':
            paths = [r['position'] for r in sorted(data['nodes'], key=lambda x: x['values']['expected_salary'], reverse=True) if not r['children_idx']][:10]
        elif return_paths_type == 'best_accumulative_salary_10':
            paths = [r for r in data['nodes'] if not r['children_idx']]
            paths = [r['position'] for r in sorted(paths, key=lambda x: x['values']['accumulative_salary'], reverse=True)][:10]
        elif return_paths_type == 'best_salary_40':
            paths = [r['position'] for r in sorted(data['nodes'], key=lambda x: x['values']['expected_salary'], reverse=True) if not r['children_idx']][:10]
        elif return_paths_type == 'best_accumulative_salary_40':
            paths = [r for r in data['nodes'] if not r['children_idx']]
            paths = [r['position'] for r in sorted(paths, key=lambda x: x['values']['accumulative_salary'], reverse=True)][:40]
        return paths
    
    try:
        dot.render(output_path, format='png', cleanup=True)
        print(f"Visualization saved as {output_path}.png")
    except Exception as e:
        print(f"Error saving visualization: {e}")
        

def process_single_tree(args):
    tree_data, model_name, i, type = args
    # try:
    paths = visualize_career_path(tree_data, f'{model_name.replace("/", "_")}_{i}', 
                                return_paths=True, return_paths_type=str(type))
    return {
        'paths': [delete_prefix(r) for r in paths],
        'reference': get_gt_from_id(tree_data['graph_id'], diversity_file),
        'unique_ratio': len(set(paths)) / len(paths)
        }
    # except Exception as e:
    #     print(f"Error processing tree {i}: {e}")
    #     return None


def run(
    type: int = 10,
    sample_n: int | None = None,
    file_start: int | None = None,
    data_start: int | None = None,
    the_most_similar: int = 5,
    file_format: str = '/scratch2/iyy1112/results/mcts_value_model/tmp_1_{}',
    filenames: list = FILENAMES
    ):
    
    diversity = Diversity('sentence-transformers/all-MiniLM-L6-v2')
    similarity = Similarity()
    
    if filenames == FILENAMES:
        if file_start != None:
            filenames = filenames[file_start:file_start+3]
            
    else:
        file_format = '{}'
        
    for model_name_or_path in filenames:
        print(f"Processing {model_name_or_path}")
        if filenames == FILENAMES:
            model_name_or_path = model_name_or_path.replace("/", "_")
        data = open_json(file_format.format(model_name_or_path))
        if sample_n:
            data = random.sample(data, sample_n)
        if data_start != None:
            data = data[data_start:data_start+3]
            
        results = []
        for i, line in tqdm(enumerate(data)):
            result = process_single_tree((line, model_name_or_path, i, type))
            if result is not None:
                results.append(result)
            
        # 결과 필터링 및 집계
        valid_results = [r for r in results if r is not None]
        
        total_paths = [r['paths'] for r in valid_results]
        references = [r['reference'] for r in valid_results]
        unique_ratios = [r['unique_ratio'] for r in valid_results]
        
        print(f"Processed {len(total_paths)} trees")
        print(f"Unique/Total: {sum(unique_ratios)/len(unique_ratios)}")
        
        # 배치 처리된 similarity 평가
        similarity_scores = similarity.group_evaluate(total_paths, references, the_most_similar)
        print(f"Similarity score: {similarity_scores}")
        diversty_scores = diversity.evaluate(total_paths)
        print(f"Diversity scores: {diversty_scores}")
        
        
if __name__ == '__main__':
    fire.Fire(run)