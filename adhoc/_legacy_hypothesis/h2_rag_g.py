from retrieval_graph import GraphDatabase, date
from sentence_transformers import SentenceTransformer
from mylmeval import open_json, get_results, save_json
import random
import fire

def run(
    model_name_or_path: str = 'google/gemma-2-2b-it',
    embedding_path: str = 'graphs_with_embedding.pkl',
    retrieval_result_path: str = 'results/baseline_retrieve/20241107_test_results.jsonl',
    do_retrieval: bool = False,
    ):
    generation_result_path = f'results/baseline_retrieve/{date}_gen_{model_name_or_path.replace("/", "_")}.jsonl'
    
    if do_retrieval:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        db = GraphDatabase(model)
        
        # 데이터 로드
        data = open_json('data/evalset/diversity.jsonl')
        
        # 그래프 ID 추출 및 분할
        graph_ids = sorted(set(item['idx'] for item in data))
        graph_len = len(graph_ids)
        
        # 그래프 단위로 train/test split
        random.seed(42)
        train_size = int(graph_len * 0.8)
        train_graph_ids = set(random.sample(graph_ids, train_size))
        test_graph_ids = set(graph_ids) - train_graph_ids
        
        print(f"Train graphs: {len(train_graph_ids)}, Test graphs: {len(test_graph_ids)}")
        
        # 그래프 처리 및 평가
        graphs, test_data = db.process_data(data, train_graph_ids, test_graph_ids, embedding_path=embedding_path)
        results = db.test(test_data, top_k=10, save_path=retrieval_result_path)
        
        
    results = open_json(retrieval_result_path)
    prompt = """Given a person in the position of {}, 
what are the 10 most recommendable career steps? 

# Similar Careerpaths, you don't need to include the exact same job titles.
{}

List only the job titles, one per line."""
    data = []
    for res in results:
        if res['query_node_id'] == 0:
            preds = [r for r in res['top_k_results']]
            preds = [p['next_nodes'] for p in preds if 'next_nodes' in p and len(p['next_nodes']) > 0]
            preds = [p[0]['content']['main'] + " " + p[0]['content'].get('detail', "") for p in preds] 
            top_k_paths = '\n'.join(preds)
            data.append({
                'inputs': [res['query'], top_k_paths],
                'metadata': {'query_graph_idx': res['query_graph_idx'], 'query_node_id': res['query_node_id']}
            })
    
    generations = get_results(
        model_name_or_path=model_name_or_path,
        data=data,
        prompt=prompt,
        save_path=generation_result_path,
        do_log=True,
        max_tokens=512,
        batch_size=len(data),
        apply_chat_template='auto',
    )
    save_json(generations, generation_result_path)
    


if __name__ == "__main__":
    fire.Fire(run)