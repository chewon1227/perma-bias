from careerpathway.scoring import Diversity, Similarity, delete_prefix, get_gt_from_id, nodes_into_string
from mylmeval import open_json
import fire
import os
from tqdm import tqdm

diversity_file = open_json('data/evalset/diversity.jsonl')

FILENAMES = [
    # 'Qwen_Qwen2.5-0.5B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-1.5B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-3B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-7B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-14B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-32B-Instruct.jsonl',
    # 'CohereForAI_aya-expanse-8b.jsonl',
    # 'CohereForAI_aya-expanse-32b.jsonl',
    # 'Qwen_Qwen2.5-72B-Instruct.jsonl',
    'gpt-4o-mini.jsonl',
    'gpt-4o.jsonl',
]


def process_file_diversity(file_path, diversity):
    example_sets = open_json(file_path)
    predictions = []
    for d in example_sets:
        preds = d['result'].split("\n")
        predictions.append(preds) # List[List[str]]
            
    print(len(predictions))
    result = diversity.evaluate(predictions, return_all=True)
    return result


def process_file_similarity(file_path, similarity, the_most_similar):
    example_sets = open_json(file_path)
    predictions = []
    references = []
    for d in tqdm(example_sets):
        results = [delete_prefix(r) for r in d['result'].split("\n")]
        predictions.append([r for r in results if len(r) > 2])
        if 'groundtruth' in d:
            references.append(nodes_into_string((d['groundtruth'])))
        else:
            references.append(get_gt_from_id(d["meta"]["graph_id"], diversity_file))
            
    result = similarity.group_evaluate(predictions, references, the_most_similar)
    return result
    

def scoring_diversity(
    file_name_format: str = 'results/eval_diversity_1/{}'
    ):
    diversity = Diversity('sentence-transformers/all-MiniLM-L6-v2')
    for result_file in FILENAMES:
        file_path = file_name_format.format(result_file)
        if os.path.exists(file_path):
            print(f"\nProcessing {result_file}")
            scores = process_file_diversity(file_path, diversity)
            print(scores)
    

def scoring_soundedness(
    file_name_format: str = 'results/eval_diversity_1/{}',
    start: int | None = None,
    the_most_similar: int = 5,
    batch_size=512
    ):
    similarity = Similarity(batch_size=batch_size)
    if start:
        filenames = FILENAMES[start:start+3]
    else:
        filenames = FILENAMES
    for result_file in filenames:
        file_path = file_name_format.format(result_file)
        if os.path.exists(file_path):
            print(f"\nProcessing {result_file}")
            scores = process_file_similarity(file_path, similarity, the_most_similar)
            print(scores)


if __name__ == '__main__':
    fire.Fire({
        'scoring_diversity' : scoring_diversity,
        'scoring_soundedness' : scoring_soundedness
    })


