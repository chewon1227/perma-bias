from careerpathway.retrieval import OptimizedRetrievalMethods
from mylmeval import open_json

data = open_json('data/evalset/diversity.jsonl')
documents = [d['content']['main']+ " " + d['content'].get('detail', '') for d in data]
queries = ['Software Developer Wavetech a/s']
retriever = OptimizedRetrievalMethods(batch_size=256)
results = retriever.semantic_search(queries, documents, top_k=100)

for result_ in results:
    for result in result_:
        instance_id = result[0]
        print(data[instance_id]['meta']['lang'])
        graph_id = data[instance_id]['idx']
        retrived = [r for r in data[instance_id+1:] if r['idx'] == graph_id]
        print(retrived)


