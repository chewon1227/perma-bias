import fire
from careerpathway.scoring.load_testset import load_diversity
from mylmeval import open_json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import torch
import os

def batch_encode(texts, model, batch_size=32):
    """Encode texts in batches"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, convert_to_tensor=True)
        embeddings.append(emb)
    return torch.cat(embeddings)

def run(
    result_path: str = 'results/eval_prompt3_100gen_2gen_2gen_2gen_2gen_2gen_2gen_2gen_0',
    batch_size = 2048,
    threshold = 0.60
    ):
    # Load data
    data = load_diversity(test_size=50)[0]
    groundtruths = [r['nodes'] for r in data]
    data = open_json(os.path.join(result_path, 'Qwen_Qwen2.5-3B-Instruct.jsonl'))
    
    # Initialize containers
    candidates = {i:[] for i in range(50)}
    stages = {i:[] for i in range(50)}
    
    # Organize data
    for i, item in enumerate(data):
        candidates[i].extend([r['content'] for r in item['nodes']])
        stages[i].extend([len(r['parent_id']) for r in item['nodes']])
    
    # Prepare model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Prepare all texts for encoding
    all_groundtruths = []
    all_candidates = []
    gt_indices = []
    cand_indices = []
    
    # Flatten and track indices
    for i, gts in enumerate(groundtruths):
        for j, gt in enumerate(gts):
            all_groundtruths.append(gt)
            gt_indices.append((i, j))
            
        for k, cand in enumerate(candidates[i]):
            all_candidates.append(cand)
            cand_indices.append((i, k))
    
    print("Encoding groundtruths...")
    gt_embeddings = batch_encode(all_groundtruths, model)
    
    print("Encoding candidates...")
    cand_embeddings = batch_encode(all_candidates, model)
    
    print("Computing similarities...")
    # Compute similarities in batches

    matches = []
    
    for i in tqdm(range(0, len(gt_embeddings), batch_size)):
        gt_batch = gt_embeddings[i:i + batch_size]
        
        # Calculate similarities with all candidates
        similarities = cosine_similarity(
            gt_batch.cpu().numpy(),
            cand_embeddings.cpu().numpy()
        )
        
        # Find matches above threshold
        for batch_idx, sim_row in enumerate(similarities):
            gt_idx = i + batch_idx
            matches.extend([
                (gt_idx, cand_idx, sim)
                for cand_idx, sim in enumerate(sim_row)
                if sim > threshold
            ])
    
    # Print results
    print("\nMatches found:", len(matches))
    for gt_idx, cand_idx, similarity in matches:
        gt_sample_idx, gt_pos = gt_indices[gt_idx]
        cand_sample_idx, cand_pos = cand_indices[cand_idx]
        
        if gt_sample_idx == cand_sample_idx and stages[cand_sample_idx][cand_pos] != 0:  # Only print matches within same sample
            print(f"Sample {gt_sample_idx}, {gt_pos}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Groundtruth: {all_groundtruths[gt_idx]}")
            print(f"Candidate: {all_candidates[cand_idx]} (Stage: {stages[cand_sample_idx][cand_pos]})")
            print('-' * 50)

if __name__ == '__main__':
    fire.Fire(run)