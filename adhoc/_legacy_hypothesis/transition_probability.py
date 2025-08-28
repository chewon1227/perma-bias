import numpy as np
import tqdm
from mylmeval.utils import open_json, save_json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from collections import defaultdict
import fire
import os
from careerpathway.scoring.load_testset import load_high_qual_diversity
from typing import List, Tuple, Dict, Set, Any, Optional, Union, DefaultDict, Iterator
import re

class CareerTransitionModel:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', 
                similarity_threshold: float = 0.9) -> None:
        # 임베딩 모델 로드
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        
        # 그래프 및 임베딩 저장소 초기화
        self.graph = nx.DiGraph()
        self.embeddings_cache = {}  # 메모리에 적재된 임베딩의 캐시
        
        # 전환 카운트 저장소
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.source_total_counts = defaultdict(int)  # 소스 노드별 총 전환 수
        
        # 노드 클러스터링을 위한 저장소
        self.node_clusters = {}  # 노드 -> 클러스터 ID
        self.cluster_members = defaultdict(list)  # 클러스터 ID -> 멤버 노드 리스트
    
    def process_profile_batches(self, profile_data_generator: Iterator[List[List[str]]], 
                               batch_size: int = 100) -> None:
        """
        대용량 프로필 데이터를 배치 단위로 처리합니다.
        
        Parameters:
        -----------
        profile_data_generator : Iterator[List[List[str]]]
            프로필 시퀀스 데이터를 생성하는 제너레이터
        batch_size : int
            한 번에 처리할 프로필 배치 크기
        """
        batch_count = 0
        
        for batch in tqdm.tqdm(profile_data_generator):
            self.add_profile_data(batch)
            
            # 메모리 관리: 임베딩 캐시 비우기
            self.embeddings_cache.clear()
            
            batch_count += 1
        
        print(f"Processed {batch_count} batches")
    
    def add_profile_data(self, profile_sequences: List[List[str]]) -> None:
        """
        프로필 시퀀스 데이터를 모델에 추가합니다.
        
        Parameters:
        -----------
        profile_sequences : List[List[str]]
            각 프로필의 커리어 시퀀스 데이터
            예: [['컴퓨터공학', 'Backend'], ['Psychology', '상담사'], ...]
        """
        # 1. 모든 노드 임베딩 생성 및 저장
        all_nodes: Set[str] = set()
        for sequence in profile_sequences:
            for node_name in sequence:
                all_nodes.add(node_name)
        
        # 새로운 노드만 임베딩 계산
        new_nodes: List[str] = [node_name for node_name in all_nodes 
                               if node_name not in self.embeddings_cache]
        
        if new_nodes:
            # 임베딩 계산 (배치 처리)
            batch_size = 32  # 더 작은 배치 사용
            for i in range(0, len(new_nodes), batch_size):
                batch = new_nodes[i:i+batch_size]
                embeddings = self.embedding_model.encode(batch)
                
                # 임베딩 저장
                for j, node_name in enumerate(batch):
                    self.embeddings_cache[node_name] = embeddings[j]
                    
                    # 그래프에 노드 추가
                    self.graph.add_node(node_name)
        
        # 2. 전환 시퀀스 분석 및 전환 카운트 업데이트
        for sequence in profile_sequences:
            for i in range(len(sequence) - 1):
                source_name = sequence[i]
                target_name = sequence[i + 1]
                
                # 직접 전환 카운트 증가
                self.transition_counts[source_name][target_name] += 1
                self.source_total_counts[source_name] += 1
                
                # 그래프에 엣지 추가 또는 가중치 업데이트
                if self.graph.has_edge(source_name, target_name):
                    self.graph[source_name][target_name]['weight'] += 1
                else:
                    self.graph.add_edge(source_name, target_name, weight=1)
        
        # 3. 유사 노드 클러스터링 (필요시 업데이트)
        self._update_node_clusters()
        
        # 4. 그래프의 전환 확률 업데이트
        self._update_transition_probabilities()
    
    def _update_node_clusters(self) -> None:
        """
        노드 임베딩을 기반으로 유사한 노드들을 클러스터링합니다.
        메모리 효율성을 위해 분할 처리됩니다.
        """
        # 임베딩이 있는 모든 노드 이름
        node_names = list(self.embeddings_cache.keys())
        
        if len(node_names) < 2:  # 클러스터링을 위한 충분한 노드가 없음
            return
        
        # 이미 클러스터링된 노드 제외
        unclustered_nodes = [n for n in node_names if n not in self.node_clusters]
        if not unclustered_nodes:
            return
            
        # 클러스터 ID 초기화
        cluster_id = max(self.cluster_members.keys(), default=-1) + 1
            
        # 메모리 효율성을 위해 배치로 처리
        batch_size = 50  # 한 번에 처리할 노드 수
        for i in range(0, len(unclustered_nodes), batch_size):
            batch_nodes = unclustered_nodes[i:i+batch_size]
            
            # 임베딩 로드
            batch_embeddings = [self.embeddings_cache[n] for n in batch_nodes]
            
            # 코사인 유사도 행렬 계산
            similarity_matrix = cosine_similarity(batch_embeddings)
            
            # 유사도 기반 클러스터링
            processed = set()
            for j, node_name in enumerate(batch_nodes):
                if node_name in processed:
                    continue
                    
                # 새 클러스터 시작
                self.node_clusters[node_name] = cluster_id
                self.cluster_members[cluster_id] = [node_name]
                processed.add(node_name)
                
                # 유사한 노드 찾기
                for k, other_name in enumerate(batch_nodes):
                    if other_name in processed:
                        continue
                        
                    # 유사도가 임계값을 넘으면 같은 클러스터에 할당
                    if similarity_matrix[j, k] >= self.similarity_threshold:
                        self.node_clusters[other_name] = cluster_id
                        self.cluster_members[cluster_id].append(other_name)
                        processed.add(other_name)
                
                cluster_id += 1
            
            # 메모리 해제
            del batch_embeddings
    
    def _update_transition_probabilities(self) -> None:
        """
        그래프의 엣지에 전환 확률을 계산하여 업데이트합니다.
        """
        for source in self.transition_counts:
            total_transitions = self.source_total_counts[source]
            if total_transitions == 0:
                continue
                
            for target in self.transition_counts[source]:
                probability = self.transition_counts[source][target] / total_transitions
                self.graph[source][target]['probability'] = probability
    
    def calculate_transition_probability(self, source_name: str, target_name: str, k: int = 5) -> float:
        # 1. 노드 임베딩이 없는 경우 계산
        if source_name not in self.embeddings_cache:
            source_embedding = self.embedding_model.encode([source_name])[0]
            self.embeddings_cache[source_name] = source_embedding
        
        if target_name not in self.embeddings_cache:
            target_embedding = self.embedding_model.encode([target_name])[0]
            self.embeddings_cache[target_name] = target_embedding
        
        # 2. 직접 전환 확률 계산
        direct_prob = 0.0
        source_pattern = re.compile(rf'.*{re.escape(source_name)}.*', re.IGNORECASE)
        target_pattern = re.compile(rf'.*{re.escape(target_name)}.*', re.IGNORECASE)

        source_keys = [key for key in self.graph if source_pattern.match(key)]
        target_keys = [key for key in self.graph if target_pattern.match(key)]
        direct_total_len = sum([len(self.graph[key]) for key in source_keys])
        
        for s_key in source_keys:
            for t_key in target_keys:
                if self.graph.has_edge(s_key, t_key):
                    direct_prob += self.graph[s_key][t_key].get('probability', 0)
        
        # 3. 유사 노드 기반 확률 추정
        similar_source_nodes = self._find_similar_nodes(source_name, k)
        similar_target_nodes = self._find_similar_nodes(target_name, k)
        
        if not similar_source_nodes or not similar_target_nodes:
            return direct_prob / direct_total_len if direct_total_len > 0 else 0.0
        
        indirect_prob = 0.0
        indirect_total_len = sum([len(self.graph[key]) for key, _ in similar_source_nodes if key in self.graph])
        for sim_source, source_sim in similar_source_nodes:
            for sim_target, target_sim in similar_target_nodes:
                if self.graph.has_edge(sim_source, sim_target):
                    edge_prob = self.graph[sim_source][sim_target].get('probability', 0)
                    # 유사도로 가중치 부여한 확률
                    indirect_prob += edge_prob * source_sim * target_sim
        
        # 4. 직접 확률과 간접 확률 결합 (가중치는 조정 가능)
        alpha = 0.7  # 직접 확률의 가중치
        if direct_total_len > 0 and indirect_total_len > 0:
            combined_prob = alpha * (direct_prob / direct_total_len) + (1 - alpha) * (indirect_prob / indirect_total_len)
        elif direct_total_len > 0:
            combined_prob = direct_prob / direct_total_len
        elif indirect_total_len > 0:
            combined_prob = indirect_prob / indirect_total_len
        else:
            combined_prob = 0.0
        
        return combined_prob
        
    def _find_similar_nodes(self, node_name: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        주어진 노드와 가장 유사한 k개의 노드를 찾습니다.
        
        Parameters:
        -----------
        node_name : str
            유사한 노드를 찾을 대상 노드 이름
        k : int
            반환할 유사 노드의 최대 수
            
        Returns:
        --------
        List[Tuple[str, float]]
            (노드이름, 유사도) 형태의 튜플 리스트, 유사도 내림차순으로 정렬됨
        """
        if node_name not in self.embeddings_cache:
            return []
            
        node_embedding = self.embeddings_cache[node_name]
        
        # 모든 노드 탐색 (노드 타입 구분 없이)
        all_nodes = [n for n in self.embeddings_cache.keys() if n != node_name]
        
        if not all_nodes:
            return []
        
        # 메모리 효율적인 방식으로 유사도 계산
        similarities = []
        
        # 배치 처리로 메모리 사용량 최적화
        batch_size = 50
        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i:i+batch_size]
            
            # 배치 임베딩 로드
            batch_embeddings = []
            for other_name in batch:
                try:
                    emb = self.embeddings_cache[other_name]
                    batch_embeddings.append((other_name, emb))
                except:
                    # 임베딩 로드 실패시 건너뛰기
                    continue
            
            # 유사도 계산
            for other_name, other_embedding in batch_embeddings:
                similarity = cosine_similarity(
                    [node_embedding], 
                    [other_embedding]
                )[0][0]
                    
                if similarity >= self.similarity_threshold:
                    similarities.append((other_name, similarity))
        
        # 유사도 내림차순으로 정렬하고 상위 k개 반환
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]


def batch_generator(data, batch_size=100):
    """데이터를 배치로 나누는 제너레이터"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + min(batch_size, len(data) - i)]

# 사용 예시 함수
def main(
    similarity_threshold: float = 0.90,
    data_batch_size: int = 500,
    test_size: int = 300,
    save_path: Optional[str] = 'results/transition_probability.jsonl'
    ) -> None:
    """
    메모리 효율적인 방식의 모델 사용 예시 데모 함수
    """
    
    if os.path.exists(save_path):
        os.remove(save_path)
        
    model = CareerTransitionModel(similarity_threshold=similarity_threshold)
    
    def run(testset: Dict[str, Any]) -> None:
        keyword = testset['keyword']
        queries = [(keyword, target) for target in testset['targets']]
        
        data, graph = load_high_qual_diversity(keyword=keyword, do_keyword=True, test_size=10000)
        data = [r['previous_nodes'] + [r['initial_node']] + r['nodes'] for r in data]
        
        data_generator = batch_generator(data, data_batch_size)
        model.process_profile_batches(data_generator)
        
        print("\n전환 확률 계산 결과:")
        print("-" * 50)
        
        for source, target in queries:
            prob = model.calculate_transition_probability(source, target)
            print(f"{source} → {target}: {prob:.4f}")
            save_json([{'source' : source, 'target' : target, 'result' : prob}], save_path, save_additionally=True)

    keywords = [
        'Computer Science',
        'Psychology',
        'Medicine',
        'Engineering',
        'Philosophy',
        'Economics',
        'Biology',
        'Art',
        'Finance',
        'Journalism',
        'Chemistry',
        'Education',
        'Music',
        'History',
        'Mathematics',
        'Physics',
        'Sociology',
        'Political Science',
        'Linguistics',
        'Translation',
    ]
    jobs = [
        'Data Scientist', 'Web Developer', 'Software Engineer', 'Product Manager', 'UX Designer', 'Machine Learning Engineer', 'Data Analyst',
        'Psychologist', 'Therapist', 'Counselor', 'Psychiatrist', 'Clinical Psychologist', 'School Psychologist', 'HR Manager',
        'Doctor', 'Nurse', 'Pharmacist', 'Medical Technologist', 'Physical Therapist', 'Dentist', 'Surgeon',
        'Civil Engineer', 'Mechanical Engineer', 'Electrical Engineer', 'Chemical Engineer', 'Aerospace Engineer', 'Architect', 'Urban Planner',
        'Philosopher', 'Ethicist', 'Theologian', 'Professor', 'Writer', 'Editor', 'Translator',
        'Economist', 'Financial Analyst', 'Accountant', 'Investment Banker', 'Actuary', 'Consultant', 'Statistician',
        'Biologist', 'Ecologist', 'Zoologist', 'Botanist', 'Geneticist', 'Microbiologist', 'Biotechnologist',
        'Artist', 'Designer', 'Illustrator', 'Animator', 'Photographer', 'Art Director', 'Curator',
        'Financial Analyst', 'Investment Banker', 'Accountant', 'Consultant', 'Trader', 'Risk Manager', 'Actuary',
        'Journalist', 'Editor', 'Reporter', 'Copywriter', 'Content Writer', 'Columnist', 'News Anchor',
        'Chemist', 'Chemical Engineer', 'Pharmacist', 'Biotechnologist', 'Material Scientist', 'Lab Technician', 'Research Scientist',
        'Teacher', 'Professor', 'Principal', 'Guidance Counselor', 'Curriculum Developer', 'Educational Psychologist', 'Librarian',
        'Musician', 'Composer', 'Conductor', 'Music Producer', 'Music Teacher', 'Music Therapist', 'Music Journalist',
        'Historian', 'Archaeologist', 'Anthropologist', 'Curator', 'Writer',
        'Mathematician', 'Statistician', 'Actuary', 'Data Scientist', 'Engineer',
        'Physicist', 'Astronomer', 'Engineer', 'Data Scientist', 'Mathematician',
        'Sociologist', 'Anthropologist', 'Social Worker', 'Policy Analyst', 'Demographer',
        'Political Scientist', 'Policy Analyst', 'Journalist', 'Lawyer', 'Diplomat',
        'Linguist', 'Translator', 'Interpreter', 'Professor', 'Researcher', 'Speech Therapist', 'Language Teacher',
    ]
    
    testset = [{'keyword': keyword, 'targets': jobs} for keyword in keywords]
    for test in testset:
        run(test)
        

if __name__ == "__main__":
    fire.Fire(main)