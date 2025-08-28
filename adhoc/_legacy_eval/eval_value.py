from mylmeval import open_json, get_results
import fire
import os
from tqdm import tqdm
from careerpathway.scoring import get_prompt_and_model

diversity_file = open_json('data/evalset/diversity.jsonl')

FILENAMES = [
    'Qwen_Qwen2.5-0.5B-Instruct.jsonl',
    'Qwen_Qwen2.5-1.5B-Instruct.jsonl',
    'Qwen_Qwen2.5-3B-Instruct.jsonl',
    'Qwen_Qwen2.5-7B-Instruct.jsonl',
    'Qwen_Qwen2.5-14B-Instruct.jsonl',
    'Qwen_Qwen2.5-32B-Instruct.jsonl',
    'CohereForAI_aya-expanse-8b.jsonl',
    # 'CohereForAI_aya-expanse-32b.jsonl',
    'Qwen_Qwen2.5-72B-Instruct.jsonl',
]

def get_data(file_name_format):
    data = []
    filenames = [f for f in FILENAMES if os.path.exists(file_name_format.format(f))]
    if file_name_format in ['results/eval_diversity/{}', 'results/eval_diversity_40/{}', 'results/baseline_retrieve/{}']: # not nodes
        for file_name in filenames:
            example_sets = open_json(file_name_format.format(file_name))
            for i, d in enumerate(example_sets):
                for re in d['result'].split("\n"):
                    data.append({
                        'inputs' : [re],
                        'meta' : {'file_name_format' : file_name_format, 'file_name' : file_name, 'idx' : i}, 
                    })
                    
    elif file_name_format in ['results/4_gar/{}', 'results/5_ragtree/{}', '/scratch2/iyy1112/results/mcts_value_model/tmp_3_{}']: # not nodes
        for file_name in filenames:
            example_sets = open_json(file_name_format.format(file_name))
            for i, d in enumerate(example_sets):
                for re in d['nodes']:
                    data.append({
                        'inputs' : [re['position']],
                        'meta' : {'file_name_format' : file_name_format, 'file_name' : file_name, 'idx' : i, 'node_id' : re['node_id']}, 
                    })
    return data                    
    
    


def run(
    file_name_format: str = 'results/eval_diversity/{}',
    value: str = 'salary', # salary, requirements,
    start: int | None = None,
    how_many: int = 1000000
    ):
    savename=file_name_format.split("/")[-2]
    prompt, model_name_or_path = get_prompt_and_model(value)
    data = get_data(file_name_format)
    if start is not None:
        data = data[start:start+how_many]
    
    os.makedirs(f'/scratch2/iyy1112/results/eval_value', exist_ok=True)
    print(len(data))
    get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=data,
        do_log=True,
        save_path=f'/scratch2/iyy1112/results/eval_value/{value}_{savename}{f"_{start}" if start!=None else ""}.jsonl',
        batch_size=len(data),
        max_tokens=512,
        apply_chat_template=True,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )

if __name__ == '__main__':
    fire.Fire(run)


