from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from careerpathway.scoring import load_diversity
from careerpathway.treesearch import CareerState
from collections import defaultdict
import numpy as np
from h1_lexical import TreeRetrieval
from mylmeval import open_json, save_json, MyLLMEval
import fire
import os
import copy
from tqdm import tqdm


class RetrievalAugmentedTreeGeneration:
    def __init__(
        self, 
        batch_size: int = 32,
        similarity_threshold: float = 0.7,
        top_k_list: List[int] = [10, 2, 2, 2]
    ):
        self.tree_retriever = TreeRetrieval(batch_size=batch_size)
        self.threshold = similarity_threshold
        self.top_k_list = top_k_list
        self.llm_generated_nodes = 0

    def convert_h1_tree_to_career_nodes(self, tree_data: List) -> List:
        for tree in tree_data:
            for node in tree['nodes']:
                if node.get('similarity', 0) < self.threshold:
                    node['h5_type'] = 'generation'
                    self.llm_generated_nodes += 1
                    for child in node['children_idx']:
                        child_node = [r for r in tree['nodes'] if r['node_id'] == child][0]
                        child_node['h5_type'] = 'generation'
                        self.llm_generated_nodes += 1

        return tree_data
        
        
    def _get_parent_history(self, node: Dict, all_nodes: List[Dict]) -> List[Tuple[str, int]]:
        history = []
        current_node = node
        while True:
            parent = None
            for potential_parent in all_nodes:
                if str(current_node['node_id']) in [str(child_id) for child_id in potential_parent.get('children_idx', [])]:
                    parent = potential_parent
                    break
            if parent is None:
                break
            history.append((parent['position'], parent['years_experience']))
            current_node = parent
        return history[::-1]
                    
                
    def fill_empty_nodes(self, trees: List[Dict], llmagent: 'MyLLMEval') -> List[Dict]:
        self.llm = llmagent
        trees = copy.deepcopy(trees)
        
        for i, top_k in enumerate(self.top_k_list):
            gen_inputs = []
            for tree in trees:
                nodes_per_years = [r for r in tree['nodes'] if r['years_experience'] == (i+1)*5]
                nodes_per_years_and_gen = [r for r in nodes_per_years if r.get('h5_type', '') == 'generation']
                if len(nodes_per_years_and_gen) == 0:
                    continue
                else:
                    gen_inputs.append({
                        'inputs' : [self._get_prompt(nodes_per_years_and_gen, tree['nodes'])],
                        'tree_id' : tree['graph_id'],
                        'which_nodes' : [r['node_id'] for r in nodes_per_years_and_gen],
                        'nodes' : nodes_per_years_and_gen
                    })

            print(f"Generating {len(gen_inputs)} nodes for level {i+1}")
            gen_outputs = self.llm.inference(
                data=gen_inputs,
                prompt="{}",
                max_tokens=100,
                batch_size=len(gen_inputs)
                )
        
            for gen_input, gen_output in zip(gen_inputs, gen_outputs):
                for gen_node, new_position in zip(gen_input['nodes'], gen_output.split("\n")):
                    gen_node['position'] = new_position
                    gen_node['h5_type'] = 'generation'
                    
        return trees
                        
    def _get_sibling(self, node: Dict, all_nodes: List[Dict]) -> int:        
        year = node['years_experience']
        node_id = node['node_id']
        return [r['position'] for r in all_nodes if r['years_experience'] == year and r['node_id'] != node_id and r.get('h5_type', '') != 'generation']

    def _get_prompt(self, nodes_to_gen: List[Dict], all_nodes: List[Dict]) -> str:
        history = self._get_parent_history(nodes_to_gen[0], all_nodes)
        career_history = "\n".join([
            f"- {position} ({years} years)"
            for position, years in history
        ])

        job_n = len(nodes_to_gen)
        year = nodes_to_gen[0]['years_experience']-5
        siblings = self._get_sibling(nodes_to_gen[0], all_nodes)
        prompt = f"""Given the career history below, suggest {job_n} different possible next job positions. Each suggestion should be a different career direction that builds upon this experience.

    Career History:
    {career_history}

    Current Profile:
    - Current Position: {history[-1][0]}
    - Years of Experience: {year}
    - Siblings: {siblings}

    Format your response with one job position per line, exactly like this example:
    Job 1
    Job 2
    Job 3

    Ensure each suggestion is realistic, distinct, and beneficial for career growth, considering the full career history."""

        return prompt



def main(
    model_name_or_path: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    threshold: float = 0.7,
    retrieval_method: str = 'semantic'
):
    # 데이터 로드
    data = open_json('data/evalset/diversity.jsonl')
    queries = load_diversity()
    documents = [item['content']['main'] + " " + item['content'].get('detail', "") 
                for item in data]
    
    ratg = RetrievalAugmentedTreeGeneration(
        similarity_threshold=threshold
    )
    if os.path.exists(f'results/1_lexical/tree_retrieval_{retrieval_method}.jsonl'):
        print("Loading existing tree retrieval results...")
        h1_trees = open_json(f'results/1_lexical/tree_retrieval_{retrieval_method}.jsonl')
        
        # Convert h1 trees to CareerNode format
        career_trees = ratg.convert_h1_tree_to_career_nodes(h1_trees) # annotate if needs generation
        print(f"Successfully loaded {len(career_trees)} trees")
        print(f"Empty nodes: {ratg.llm_generated_nodes}, Generation ratio: {ratg.llm_generated_nodes/(len(career_trees)*80)}")
            
    else:
        print("No existing tree retrieval results found. Creating new trees...")
        ratg = RetrievalAugmentedTreeGeneration(
            similarity_threshold=threshold
        )
        
        career_trees = ratg.build_career_trees(
            documents=documents,
            queries=queries,
            method=retrieval_method
        )
        
        print(f"Built {len(career_trees)} trees")
        
    
    llm_client = MyLLMEval(model_name_or_path)
    completed_trees = ratg.fill_empty_nodes(career_trees, llm_client)
    
    os.makedirs(f'results/5_ragtree_{threshold}', exist_ok=True)
        
    save_json(completed_trees, f'results/5_ragtree_{threshold}/{model_name_or_path.replace("/", "_")}.jsonl')


if __name__ == "__main__":
    fire.Fire(main)

