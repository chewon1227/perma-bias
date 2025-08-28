from typing import List, Tuple, Dict
from collections import defaultdict
from careerpathway.retrieval import OptimizedRetrievalMethods
from careerpathway.scoring import load_diversity, load_issue, content_into_string
import fire
from mylmeval import open_json, save_json


class TreeRetrieval:
    def __init__(self, batch_size: int = 32, next_node_fn: callable = None):
        self.batch_size = batch_size
        self.next_node_fn = next_node_fn
        self.retriever = OptimizedRetrievalMethods(batch_size=self.batch_size)

    def retrieve_single(
        self,
        method: str,
        documents: List[str],
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[int, float, str]]]:
        print(f"Using device: {self.retriever.device}")
        print(f"Processing {len(queries)} queries in batches of {self.batch_size}")
        
        if method == 'semantic':
            results = self.retriever.semantic_search(queries, documents, top_k=top_k)
        elif method == 'lexical':
            results = self.retriever.lexical_search(queries, documents, top_k=top_k)
        elif method == 'hybrid':
            results = self.retriever.hybrid_search(queries, documents, top_k=top_k)
        else:
            raise ValueError(f"Unknown method: {method}")
        return results
    
    
    def retrieve_multi_multi(
        self,
        method: str,
        documents: List[List],
        queries: List[List],
        top_k: int = 10,
        mean_k: int = 20
    ) -> List[List[Tuple[int, float, str]]]:
        print(f"Using device: {self.retriever.device}")
        print(f"Processing {len(queries)} queries in batches of {self.batch_size}")

        total_result = []
        for query_list in queries:
            retrieved = []
            for doc_i, document_list in enumerate(documents):
                if method == 'semantic':
                    results = self.retriever.semantic_search(query_list, document_list, top_k=min(len(document_list), mean_k)-1)
                elif method == 'lexical':
                    results = self.retriever.lexical_search(query_list, document_list, top_k=min(len(document_list), mean_k)-1)
                elif method == 'hybrid':
                    results = self.retriever.hybrid_search(query_list, document_list, top_k=min(len(document_list), mean_k)-1) 
                else:
                    raise ValueError(f"Unknown method: {method}")
                mean_score = sum([r[1] for r in results[0]]) / len(results[0]) if len(results[0]) > 0 else 0
                retrieved.append((doc_i, mean_score, document_list))
            total_result.append(sorted(retrieved, key=lambda x: x[1], reverse=True)[:top_k])
        return total_result
        

    def initialize_tree(self, query_item: Dict) -> Dict:
        """각 쿼리에 대한 트리 구조를 초기화합니다."""
        tree = {
            "graph_id": query_item['graph_id'],
            "initial_node": query_item['initial_node'],
            "nodes": []
        }
        
        # 초기 노드 추가
        initial_node = {
            "position": query_item['initial_node'],
            "years_experience": 0,
            "children_idx": [],
            "node_id": 0
        }
        tree["nodes"].append(initial_node)
        
        return tree

    def add_node_to_tree(
        self,
        tree: Dict,
        parent_idx: int,
        next_position: str,
        level: int,
        score: float
    ) -> int:
        """트리에 새로운 노드를 추가하고 노드 ID를 반환합니다."""
        new_node_id = len(tree["nodes"])
        
        # 새 노드 생성
        node = {
            "position": next_position,
            "years_experience": (level + 1) * 5,
            "children_idx": [],
            "node_id": new_node_id,
            'similarity': score
        }
        
        # 부모 노드의 children_idx에 새 노드 ID 추가
        tree["nodes"][parent_idx]["children_idx"].append(new_node_id)
        
        # 트리에 새 노드 추가
        tree["nodes"].append(node)
        return new_node_id

    def retrieve_tree(
        self,
        documents: List[str],
        queries: List[dict],
        top_k_list: List[int] = [10, 2, 2, 2],
        method: str = 'semantic',
    ) -> List[Dict]:
        print(f"Using device: {self.retriever.device}")
        
        # 트리 초기화
        trees = [self.initialize_tree(query_item) for query_item in queries]
        
        # 현재 레벨의 쿼리들 - (query_idx, current_node_id, position) 튜플로 관리
        all_current_queries = [(i, 0, query_item['initial_node']) 
                             for i, query_item in enumerate(queries)]
        
        for level, top_k in enumerate(top_k_list):
            print(f"Processing level {level} with {len(all_current_queries)} queries")
            
            # 현재 쿼리 텍스트만 추출
            current_query_texts = [q[2] for q in all_current_queries]
            
            results = self.retrieve_single(
                method=method,
                queries=current_query_texts,
                documents=documents,
                top_k=100,
            )
            
            next_queries = []
            for (query_idx, current_node_id, _), query_results in zip(all_current_queries, results):
                currently_added = 0
                for doc_idx, score, doc in query_results:
                    if currently_added >= top_k:
                        break
                        
                    next_position = self.next_node_fn(doc_idx, doc)
                    if next_position is None:
                        continue
                        
                    new_node_id = self.add_node_to_tree(
                        tree=trees[query_idx],
                        parent_idx=current_node_id,
                        next_position=next_position,
                        level=level,
                        score=score
                    )
                    
                    currently_added += 1
                    next_queries.append((query_idx, new_node_id, next_position))
                    
            save_json(trees, f'results/1_lexical/tmp_{method}_{level}.jsonl', save_additionally=True)
            all_current_queries = next_queries
        
        return trees


def main(
    method: str = 'semantic',
    batch_size: int = 2048,
    start: int | None = None,
    input_type: str = 'issue'
    ):
    
    def next_step_in_graph(doc_idx: int, doc: str) -> str:
        graph_id = data[doc_idx]['idx']
        nodes = [r['content']['main'] + " " + r['content'].get('detail', '') for r in graphs[graph_id]]
        now_idx = nodes.index(doc)
        return nodes[now_idx + 1] if now_idx + 1 < len(nodes) else None
    
    data = open_json('data/evalset/diversity.jsonl')
    documents = [content_into_string(r['content']) for r in data]
    testset, graphs = load_diversity()
    
    if input_type == 'issue':
        testset, _ = load_issue()
        
    retriever = TreeRetrieval(batch_size=batch_size, next_node_fn=next_step_in_graph)
    
    print("\nTesting tree retrieval:")
    trees = retriever.retrieve_tree(
        documents=documents,
        queries=testset[start:start+2000] if start != None else testset,
        top_k_list=[10, 2, 2, 2],
        method=method,
    )
    
    save_json(trees, f'results/1_lexical/tree_retrieval_{method}_{start}_{input_type}.jsonl')


if __name__ == "__main__":
    fire.Fire(main)