from mylmeval import open_json
from careerpathway.retrieval import OptimizedRetrievalMethods


def test(method: str, top_k: int = 15, batch_size: int = 32):
    retriever = OptimizedRetrievalMethods(batch_size=batch_size)
    data = open_json('data/evalset/diversity.jsonl')
    documents = [item['content']['main'] + " " + item['content'].get('detail', "") for item in data]
    queries = [item['content']['main'] for item in data[:64]]
    
    print(f"Using device: {retriever.device}")
    print(f"Processing {len(queries)} queries in batches of {batch_size}")
    
    if method == 'semantic':
        results = retriever.semantic_search(queries, documents, top_k=top_k)
    elif method == 'lexical':
        results = retriever.lexical_bm25_search(queries, documents, top_k=top_k)
    elif method == 'hybrid':
        results = retriever.hybrid_search(queries, documents, top_k=top_k)
    
    # 결과 출력
    for query, query_results in zip(queries, results):
        print(f"Query: {query}, Method: {method}")
        print('\n'.join([f"Score: {score:.2f}, Index: {idx}, Document: {doc[:100]}..." for idx, score, doc in query_results]))
        print('---')

    
if __name__ == "__main__":
    batch_size = 64  # GPU 메모리에 따라 조정
    for method in ['semantic', 'lexical']:
        test(method, batch_size=batch_size)