import random
import json
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
from typing import List, Dict, Any, Tuple, Set
import fire
from sklearn.model_selection import train_test_split
from mylmeval import open_json, save_json
import os
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import datetime
date = datetime.datetime.now().strftime("%Y%m%d")


class GraphDatabase:
    def __init__(self, model):
        self.model = model
        self.graphs = []  # List of individual graphs
        self.graph_id_to_idx = {}  # 그래프 ID -> 인덱스 매핑
        
    def process_data(self, graph_data: Dict[int, List], train_graph_ids: List[int], 
                    test_graph_ids: List[int], embedding_path: str = None) -> Tuple[List[nx.DiGraph], List[Dict]]:

        # 학습용 데이터만 선택
        train_data = []
        for graph_id in train_graph_ids:
            if graph_id in graph_data:
                train_data.extend(graph_data[graph_id])
        
        # 테스트용 데이터 선택
        test_data = []
        for graph_id in test_graph_ids:
            if graph_id in graph_data:
                test_data.extend(graph_data[graph_id])
                
        # 학습용 그래프 생성
        if embedding_path and os.path.exists(embedding_path):
            try:
                self.graphs = self.load_graphs(embedding_path)
                if self._validate_graphs(train_graph_ids):
                    print(f"Successfully loaded valid graphs from {embedding_path}")
                    return self.graphs, test_data
                print("Loaded graphs are invalid or missing embeddings, rebuilding...")
            except Exception as e:
                print(f"Error loading saved graphs: {e}, rebuilding...")
        
        return self._build_graphs(train_data, embedding_path), test_data
    
    def _validate_graphs(self, expected_graph_ids: Set[int]) -> bool:
        """
        로드된 그래프의 유효성을 검증합니다.
        """
        try:
            # 그래프 ID 확인
            loaded_graph_ids = {graph.graph['graph_idx'] for graph in self.graphs}
            if loaded_graph_ids != expected_graph_ids:
                print("Mismatch in graph IDs")
                return False
            
            # 각 그래프의 노드 검증
            for graph in self.graphs:
                for node in graph.nodes():
                    if 'embedding' not in graph.nodes[node]:
                        print(f"Missing embedding for node {node} in graph {graph.graph['graph_idx']}")
                        return False
            return True
            
        except Exception as e:
            print(f"Error during graph validation: {e}")
            return False
    
    def _build_graphs(self, train_data: List[Dict], save_path: str = None) -> List[nx.DiGraph]:
        """
        학습 데이터로부터 그래프를 구축합니다.
        """
        # 임베딩 생성
        contents = [
            f"{item['content']['main']} {item['content'].get('detail', '')}"
            for item in train_data
        ]
        
        embeddings = self.make_embedding(contents)
        if len(embeddings) != len(train_data):
            raise ValueError(f"임베딩 수 불일치: 예상 {len(train_data)}, 실제 {len(embeddings)}")
        
        # 그래프 구축
        graph_dict = {}  # graph_id -> DiGraph
        for item, embedding in zip(train_data, embeddings):
            graph_id = item['idx']
            
            if graph_id not in graph_dict:
                G = nx.DiGraph()
                G.graph['graph_idx'] = graph_id
                graph_dict[graph_id] = G
            
            content = f"{item['content']['main']} {item['content'].get('detail', '')}"
            
            # 노드 추가
            graph_dict[graph_id].add_node(
                item['node'],
                content=content,
                raw_content=item['content'],
                graph_idx=graph_id,
                embedding=embedding
            )
            
            # 엣지 추가
            if item['from'] is not None:
                graph_dict[graph_id].add_edge(item['from'], item['node'])
            if item['to'] is not None:
                graph_dict[graph_id].add_edge(item['node'], item['to'])
        
        self.graphs = list(graph_dict.values())
        
        # 그래프 저장
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(self.graphs, f)
            print(f"Saved {len(self.graphs)} graphs to {save_path}")
        
        return self.graphs
    
    def load_graphs(self, load_path: str) -> List[nx.DiGraph]:
        """저장된 그래프를 로드합니다."""
        with open(load_path, 'rb') as f:
            graphs = pickle.load(f)
        print(f"Loaded {len(graphs)} graphs from {load_path}")
        return graphs
    
    def make_embedding(self, contents: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """컨텐츠의 임베딩을 생성합니다."""
        embeddings = []
        processed_contents = [
            str(content).strip() if content is not None else ""
            for content in contents
        ]
        
        for i in tqdm(range(0, len(processed_contents), batch_size), desc="Generating embeddings"):
            batch = processed_contents[i:i + batch_size]
            batch_embeddings = self.model.encode(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def prepare_node_embeddings(self):
        """모든 노드의 임베딩을 하나의 행렬로 준비"""
        self.all_embeddings = []
        self.node_info = []  # [(graph_idx, node_id), ...]
        
        for graph in tqdm(self.graphs):
            graph_idx = graph.graph['graph_idx']
            for node in graph.nodes():
                node_data = graph.nodes[node]
                if 'embedding' not in node_data:
                    print(f"Warning: Node {node} in graph {graph_idx} missing embedding")
                    continue
                self.all_embeddings.append(node_data['embedding'])
                self.node_info.append((graph_idx, node))
        
        self.all_embeddings = np.array(self.all_embeddings)

    def batch_find_similar_nodes(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        """배치로 유사한 노드들을 찾습니다."""
        # 쿼리 임베딩을 배치로 생성
        query_embeddings = self.model.encode(queries)
        
        # 배치 유사도 계산 (num_queries x num_nodes)
        similarities = np.dot(query_embeddings, self.all_embeddings.T)
        
        # 각 쿼리에 대한 결과 처리
        all_results = []
        for query_idx, query_similarities in enumerate(similarities):
            # top-k 인덱스 찾기
            top_indices = np.argsort(query_similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                graph_idx, node = self.node_info[idx]
                similarity = float(query_similarities[idx])
                
                # 해당 그래프 찾기
                graph = next(g for g in self.graphs if g.graph['graph_idx'] == graph_idx)
                
                # 다음 노드들 찾기
                next_nodes = list(graph.successors(node))
                
                result = {
                    'query_node': {
                        'id': node,
                        'content': graph.nodes[node]['raw_content'],
                        'similarity': similarity
                    },
                    'next_nodes': [{
                        'id': next_node,
                        'content': graph.nodes[next_node]['raw_content']
                    } for next_node in next_nodes],
                    'graph_idx': graph_idx
                }
                results.append(result)
                
            all_results.append(results)
        
        return all_results

    def test(self, test_data: List[Dict], save_path: str, top_k: int = 5, batch_size: int = 32) -> List[Dict[str, Any]]:
        # 노드 임베딩 행렬 준비
        os.makedirs('results/baseline_retrieve', exist_ok=True)
        self.save_path = save_path
            
        if not hasattr(self, 'all_embeddings'):
            print("Preparing node embeddings...")
            self.prepare_node_embeddings()
        
        results = []
        skipped = 0
        
        # 배치 처리
        queries = []
        query_metas = []
        
        for data in test_data:
            query = f"{data['content']['main']} {data['content'].get('detail', '')}"
            queries.append(query)
            query_metas.append({
                'query': query,
                'query_graph_idx': data['idx'],
                'query_node_id': data['node']
            })
            
            # 배치 크기에 도달하면 처리
            if len(queries) >= batch_size:
                batch_results = self.batch_find_similar_nodes(queries, top_k)
                
                for query_meta, similar_nodes in zip(query_metas, batch_results):
                    if not similar_nodes:
                        skipped += 1
                        continue
                        
                    results.append({
                        **query_meta,
                        'top_k_results': similar_nodes
                    })
                
                queries = []
                query_metas = []
                
                # 중간 결과 저장
                if len(results) % 1000 == 0:
                    save_json(results, self.save_path)
                    print(f"Processed {len(results)} queries, skipped {skipped}")
        
        # 남은 쿼리 처리
        if queries:
            batch_results = self.batch_find_similar_nodes(queries, top_k)
            for query_meta, similar_nodes in zip(query_metas, batch_results):
                if not similar_nodes:
                    skipped += 1
                    continue
                    
                results.append({
                    **query_meta,
                    'top_k_results': similar_nodes
                })
        
        print(f"Completed testing. Processed {len(results)} queries, skipped {skipped}")
        save_json(results, self.save_path)
        return results


def run(
    embedding_path: str = 'graphs_with_embedding.pkl',
    save_path: str = f'results/baseline_retrieve/{date}_test_results.jsonl',
    ):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    db = GraphDatabase(model)
    
    # 데이터 로드
    data = open_json('data/evalset/diversity.jsonl')

    data_graphs = defaultdict(list)    
    graph_ids = sorted(set(item['idx'] for item in data))
    for item in data:
        data_graphs[item['idx']].append(item)
        
    test_graph_ids = [k for k, v in data_graphs.items() if len(v) == 1]
    train_graph_ids = [k for k in graph_ids if k not in test_graph_ids]
    
    print(f"Train graphs: {len(train_graph_ids)}, Test graphs: {len(test_graph_ids)}")
    
    # 그래프 처리 및 평가
    graphs, test_data = db.process_data(data_graphs, train_graph_ids, test_graph_ids, embedding_path=embedding_path)
    results = db.test(test_data, top_k=10, save_path=save_path)
    


if __name__ == "__main__":
    fire.Fire(run)