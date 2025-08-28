from typing import List, Tuple, Dict
from collections import defaultdict
from careerpathway.scoring import load_diversity, load_issue, process_str_to_answers, content_into_string
from mylmeval import open_json, save_json
import fire
import os
from h1_lexical import TreeRetrieval
from treebase import Tree


    
def main(
    model_name_or_path: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    method: str = 'semantic',
    start: int | None = None,
    input_type: str = 'diversity',
    batch_size: int = 64
    ):
    # 데이터 로드
    data = open_json('data/evalset/diversity.jsonl')
    documents = [item['content']['main'] + " " + item['content'].get('detail', "") for item in data]
    
    def next_step_in_graph(doc_idx: int, doc: str) -> str:
        graph_id = data[doc_idx]['idx']
        nodes = [r['content']['main'] + " " + r['content'].get('detail', '') for r in graphs[graph_id]]
        now_idx = nodes.index(doc)
        return nodes[now_idx + 1] if now_idx + 1 < len(nodes) else None
    
    queries, graphs = load_diversity(test_size=10000)
    if input_type == 'issue':
        queries, _ = load_issue()
    pipeline = Tree(batch_size=batch_size, model_name_or_path=model_name_or_path, method=method, next_node_fn=next_step_in_graph)

    trees = pipeline.retrieve_tree(
        documents=documents,
        queries=queries[start:start+2000] if start is not None else queries,
        top_k_list=[10, 2, 2, 2],
        method=method
    )

    os.makedirs('results/4_gar', exist_ok=True)
    save_json(trees, f'results/4_gar/{model_name_or_path.replace("/", "_")}{f"_{start}" if start != None else ""}.jsonl')
 

if __name__ == "__main__":
    fire.Fire(main)