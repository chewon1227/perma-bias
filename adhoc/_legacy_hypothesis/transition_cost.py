import re
import fire
import tqdm
import random
import numpy as np
import networkx as nx
from mylmeval.utils import open_json, save_json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from itertools import combinations
from typing import List, Tuple, Dict, Set, Any, Optional, Union, DefaultDict, Iterator

class TransitionCostModel:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', 
                similarity_threshold: float = 0.9) -> None:
        # 임베딩 모델 로드
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.similarity_threshold = similarity_threshold
        
        # 그래프 및 임베딩 저장소 초기화
        self.embeddings_cache = {}  # 메모리에 적재된 임베딩의 캐시
        
        # 전환 카운트 저장소
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.source_total_counts = defaultdict(int)  # 소스 노드별 총 전환 수
        
        # 노드 클러스터링을 위한 저장소
        self.node_clusters = {}  # 노드 -> 클러스터 ID
        self.cluster_members = defaultdict(list)  # 클러스터 ID -> 멤버 노드 리스트
        
        # 요구사항 데이터 저장
        self.requirement_data = []
    
    def _calculate_transition_cost_text(self, sources: List[str], targets: List[str]) -> float:
        """
        여러 소스와 타겟 간의 텍스트 기반 요구사항 차집합을 계산하여 전환 비용을 반환합니다.
        
        Parameters:
        -----------
        sources : List[str]
            출발점 직업/전공 이름 리스트
        targets : List[str]
            도착점 직업/전공 이름 리스트
            
        Returns:
        --------
        float
            텍스트 기반 전환 비용 (0~1 사이의 값, 1에 가까울수록 전환이 어려움)
        """
        # 소스와 타겟에 대한 요구사항 집합 생성
        source_reqs = set()
        target_reqs = set()
        sources = list(set(sources))
        targets = list(set(targets))
        
        # 정규식 패턴으로 관련 직업/전공 찾기
        for source in tqdm.tqdm(sources, desc="Collecting source requirements: text"):
            pattern = re.compile(f".*{re.escape(source)}.*", re.IGNORECASE)
            for item in self.requirement_data:
                if pattern.match(item.get('job', '')) and item.get('skills', []):
                    source_reqs.update(item.get('skills', []))
        
        for target in tqdm.tqdm(targets, desc="Collecting target requirements: text"):
            pattern = re.compile(f".*{re.escape(target)}.*", re.IGNORECASE)
            for item in self.requirement_data:
                if pattern.match(item.get('job', '')) and item.get('skills', []):
                    target_reqs.update(item.get('skills', []))
        
        # 요구사항이 없는 경우 최대 비용 반환
        if not source_reqs or not target_reqs:
            return target_reqs, source_reqs, 1.0
        
        # 타겟에만 있는 요구사항 계산 (차집합)
        additional_reqs = target_reqs - source_reqs
        
        # 전환 비용 계산: 추가 요구사항 / 타겟 총 요구사항
        if not target_reqs:
            return target_reqs, source_reqs, 0

        print(f"Additional requirements: {len(additional_reqs)}")        
        return target_reqs, source_reqs, additional_reqs

    def _calculate_transition_cost_emb(self, sources: List[str], targets: List[str]) -> float:
        """
        여러 소스와 타겟 사이의 임베딩 기반 거리를 계산합니다.
        
        Parameters:
        -----------
        sources : List[str]
            출발점 직업/전공 이름 리스트
        targets : List[str]
            도착점 직업/전공 이름 리스트
            
        Returns:
        --------
        float
            임베딩 기반 전환 비용 (0~1 사이의 값, 1에 가까울수록 전환이 어려움)
        """
        # 소스와 타겟 요구사항 수집
        source_skills = []
        target_skills = []
        
        # 정규식 패턴으로 관련 직업/전공 찾기
        for source in tqdm.tqdm(sources, desc="Collecting source requirements: embedding"):
            pattern = re.compile(f".*{re.escape(source)}.*", re.IGNORECASE)
            for item in self.requirement_data:
                if pattern.match(item.get('job', '')) and item.get('skills', []):
                    source_skills.extend(item.get('skills', []))
        
        for target in tqdm.tqdm(targets, desc="Collecting target requirements: embedding"):
            pattern = re.compile(f".*{re.escape(target)}.*", re.IGNORECASE)
            for item in self.requirement_data:
                if pattern.match(item.get('job', '')):
                    if item.get('skills', []):
                        target_skills.extend(item['skills'])
        
        # 소스와 타겟 요구사항 텍스트 생성
        source_text = " ".join(source_skills)
        target_text = " ".join(target_skills)
        
        # 요구사항이 없는 경우 최대 비용 반환
        if not source_text or not target_text:
            return 1.0
        
        # 소스와 타겟의 임베딩 생성
        source_embedding = self.embedding_model.encode(source_text)
        target_embedding = self.embedding_model.encode(target_text)
        
        # 임베딩 간 코사인 유사도 계산
        similarity = cosine_similarity(
            [source_embedding], 
            [target_embedding]
        )[0][0]
        
        # 유사도를 거리(비용)로 변환 (1 - 유사도)
        return 1.0 - similarity


    def upload_essential_data(self, data: List[Dict[str, str]], queries: List[Tuple[str, str]], do_emb_cache: bool = False, do_subset: bool = True, subset_size: int = 5000) -> None:
        # 요구사항 데이터 저장
        self.requirement_data = data
        essential_nodes = list(set([target for _, target in queries] + [source for source, _ in queries]))
        
        # 정규식으로 범위를 넓혀 관련 노드 찾기
        essential_jobs = []
        for node in essential_nodes:
            pattern = re.compile(f".*{re.escape(node)}.*", re.IGNORECASE)
            related_jobs = [(item.get('job', ''), item.get('skills', [])) 
                           for item in self.requirement_data 
                           if pattern.match(item.get('job', ''))]
            essential_jobs.extend(related_jobs)
            
        if do_subset:
            essential_jobs = random.sample(essential_jobs, min(subset_size, len(essential_jobs)))
        
        if do_emb_cache:
            print(f"요구사항 임베딩 계산 중... {len(essential_jobs)}")
            batch_size = 1000
            self.embeddings_cache = {}
            
            for batch in tqdm.tqdm(batch_generator(essential_jobs, batch_size)):
                for name, requirements in batch:
                    if not requirements:
                        continue
                        
                    # 요구사항을 하나의 텍스트로 결합
                    combined_text = " ".join(requirements)
                    
                    # 임베딩 계산 및 캐싱
                    self.embeddings_cache[name] = self.embedding_model.encode(combined_text)
            
            print(f"총 {len(self.embeddings_cache)} 개의 임베딩이 계산되었습니다.")
        
    def calculate_transition_cost(self, source: str, target: str, do_sample: bool=True, sample_size: int=500) -> Tuple[float, float]:
        # 정규식으로 범위를 넓혀 관련 노드 찾기
        source_pattern = re.compile(f".*{re.escape(source)}.*", re.IGNORECASE)
        target_pattern = re.compile(f".*{re.escape(target)}.*", re.IGNORECASE)
        
        sources = [item.get('job', '') for item in self.requirement_data 
                  if source_pattern.match(item.get('job', ''))]
        targets = [item.get('job', '') for item in self.requirement_data 
                  if target_pattern.match(item.get('job', ''))]
        
        if not sources:
            sources = [source]
        if not targets:
            targets = [target]
            
        if do_sample:
            sources = random.sample(sources, min(sample_size, len(sources)))
            targets = random.sample(targets, min(sample_size, len(targets)))
        
        source_req, target_req, text_cost = self._calculate_transition_cost_text(sources, targets)
        emb_cost = self._calculate_transition_cost_emb(sources, targets)
        
        return text_cost, emb_cost


def batch_generator(data, batch_size=100):
    """데이터를 배치로 나누는 제너레이터"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + min(batch_size, len(data) - i)]

# 사용 예시 함수
def main(
    similarity_threshold: float = 0.90,
    requirement_data_path: str = 'data/data13_15_kaggle.jsonl',
    save_path: str = 'results/transition_cost.jsonl',
    start: int = 0,
    end: int | None = None
    ) -> None:
    """
    메모리 효율적인 방식의 모델 사용 예시 데모 함수
    """
    
    model = TransitionCostModel(similarity_threshold=similarity_threshold)
    
    def run(testset: List[Dict[str, str]]) -> None:
        queries = [(item['keyword'], item['targets']) for item in testset]
        
        data = open_json(requirement_data_path)
        data = [{'job' : r['job'], 'skills' : r['skills']} for r in data]
        model.upload_essential_data(data, queries)
        
        for source, target in queries:
            try:
                cost_text, cost_emb = model.calculate_transition_cost(source, target, sample_size=100)
                print(f"{source} → {target}: text: {list(cost_text)[:10]}, emb: {cost_emb:.4f}")
                save_json([{'source' : source, 'target' : target, 'result' : cost_emb}], save_path, save_additionally=True)
                
            except Exception as e:
                print(f"Error: {e}")
        
    
    # testset = [
    #     {'keyword' : 'Software Engineer', 'targets' : 'Counselor'},
    #     {'keyword' : 'Psychologist', 'targets' : 'Counselor'},
    #     {'keyword' : 'Counselor', 'targets' : 'Software Engineer'},
    #     {'keyword' : 'Accountant', 'targets' : 'Software Engineer'}
    # ]
    jobs = [
        'Software Engineer', 'Frontend Engineer', 'Backend Engineer', 'Data Scientist', 'Data Analyst', 'Machine Learning Engineer',
        'Psychologist', 'Counselor', 'Therapist', 'Psychiatrist', 'Social Worker', 'Mental Health Counselor',
        'Accountant', 'Financial Analyst', 'Investment Banker', 'Tax Accountant',
        'Doctor', 'Nurse', 'Medical Assistant', 'Surgeon', 'Anesthesiologist', 'Pediatrician',
        'Teacher', 'Professor', 'Principal', 'School Counselor', 'Librarian', 'Curriculum Developer'
    ]
    job_combination = combinations(jobs, 2)
    testset = [{'keyword' : job1, 'targets' : job2} for job1, job2 in job_combination]
    print(f"Total {len(testset)} test cases are loaded.")
    end = end or start + 50
    run(testset[start:end])
        

if __name__ == "__main__":
    fire.Fire(main)