from mylmeval import open_json, MyLLMEval
import os, fire
import json
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from mylmeval import open_json, MyLLMEval
import os, fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

class DiversityBeamSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.lambda_param = 0.7

    def get_embeddings(self, texts):
        return self.encoder.encode(texts, normalize_embeddings=True)

    def compute_similarity_matrix(self, embeddings):
        n = len(embeddings)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                sim_matrix[i][j] = 1 - cosine(embeddings[i], embeddings[j])
        return sim_matrix

    def mmr_selection(self, candidate_embeddings, candidates, top_k=10):
        if len(candidates) <= top_k:
            return list(range(len(candidates)))

        sim_matrix = self.compute_similarity_matrix(candidate_embeddings)
        avg_sims = np.mean(sim_matrix, axis=1)
        selected_idx = [np.argmax(avg_sims)]
        unselected = list(range(len(candidates)))
        unselected.remove(selected_idx[0])

        while len(selected_idx) < top_k:
            mmr_scores = []
            for i in unselected:
                similarities = [sim_matrix[i][j] for j in selected_idx]
                max_sim = max(similarities)
                relevance = np.mean(sim_matrix[i])
                diversity = -max_sim
                mmr_score = (1 - self.lambda_param) * relevance + self.lambda_param * diversity
                mmr_scores.append(mmr_score)

            next_idx = unselected[np.argmax(mmr_scores)]
            selected_idx.append(next_idx)
            unselected.remove(next_idx)

        return selected_idx




class HierarchicalBeamSearch:
    def __init__(self, model_path='Qwen/Qwen2.5-3B-Instruct'):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval()
    
    def compute_path_probability(self, parent_text, child_text):
        """부모-자식 경로의 확률 계산"""
        sequence = f"{parent_text} → {child_text}"
        inputs = self.tokenizer(sequence, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 토큰별 확률 계산
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            probs = torch.softmax(shift_logits, dim=-1)
            token_probs = torch.gather(
                probs, 2, 
                shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # 화살표 이후의 확률만 고려
            arrow_token = self.tokenizer.encode(" →", add_special_tokens=False)[0]
            arrow_pos = (inputs['input_ids'] == arrow_token).nonzero(as_tuple=True)[1]
            
            if len(arrow_pos) > 0:
                transition_probs = token_probs[0, arrow_pos[0]:]
                path_prob = torch.exp(torch.mean(torch.log(transition_probs)))
            else:
                path_prob = torch.mean(token_probs)
                
        return path_prob.item()
    
    def select_best_paths(self, current_nodes, next_stage_nodes, beam_size=10):
        """모든 가능한 경로에서 가장 확률이 높은 beam_size개 선택"""
        all_paths = []  # (parent_idx, child_idx, probability, parent_node, child_node)
        
        # 각 부모 노드에서 모든 자식 노드로의 경로 확률 계산
        for parent_idx, parent_node in enumerate(current_nodes):
            for child_idx, child_node in enumerate(next_stage_nodes):
                prob = self.compute_path_probability(parent_node['content'], child_node['content'])
                all_paths.append((parent_idx, child_idx, prob, parent_node, child_node))
        
        # 확률로 정렬하고 상위 beam_size개 선택
        sorted_paths = sorted(all_paths, key=lambda x: x[2], reverse=True)
        best_paths = sorted_paths[:beam_size]
        
        # 선택된 자식 노드들의 parent_id 업데이트
        selected_nodes = []
        for _, _, prob, parent_node, child_node in best_paths:
            child_node['parent_id'] = parent_node.get('parent_id', []) + [parent_node['node_id']]
            child_node['path_probability'] = prob
            selected_nodes.append(child_node)
        
        return selected_nodes, best_paths

def run(
    file_path: str = 'results/eval_us_M_prompt3_10gen_2gen_2gen_2gen_2gen_0',
    beam_size: int = 10,
    model_path: str = 'Qwen/Qwen2.5-3B-Instruct'
):
    data = open_json(os.path.join(file_path, 'Qwen_Qwen2.5-3B-Instruct.jsonl'))
    beam_search = HierarchicalBeamSearch(model_path)
    
    for item in tqdm(data):
        print('Initial node:', item['initial_node'])
        initial_node = {'content': item['initial_node'], 'node_id': 'initial'}
        nodes = item['nodes']
        current_nodes = [initial_node]
        
        # 각 스테이지별로 처리
        for stage in range(0, file_path.count('gen')):
            print(f"\nProcessing Stage {stage}")
            
            # 현재 스테이지의 노드들 가져오기
            stage_nodes = [n for n in nodes if len(n.get('parent_id', [])) == stage]
            print(f"Total candidates for stage {stage}: {len(stage_nodes)}")
            
            if len(stage_nodes) > 0:
                # 확률 기반으로 최적의 경로 선택
                selected_nodes, best_paths = beam_search.select_best_paths(
                    current_nodes,
                    stage_nodes,
                    beam_size
                )
                
                # 결과 출력
                print(f"\nTop {beam_size} paths selected for stage {stage}:")
                for _, _, prob, parent, child in best_paths:
                    print(f"{parent['content']} → {child['content']}: {prob:.6f}")
                
                # 다음 스테이지를 위해 현재 선택된 노드들 저장
                current_nodes = selected_nodes
                
                # 원본 nodes 리스트 업데이트
                nodes = [n for n in nodes if n not in stage_nodes] + selected_nodes
        
        # 최종 선택된 노드들로 트리 업데이트
        item['nodes'] = nodes
    
    # 결과 저장
    output_path = os.path.join(file_path, f'Qwen_Qwen2.5-3B-Instruct_hierarchical_beam{beam_size}.jsonl')
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    fire.Fire(run)
    fire.Fire(run)