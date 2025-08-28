from mylmeval import open_json, save_json, get_results
import random
import pandas as pd
from collections import Counter
from careerpathway.utils import extract_num

result_paths = [
    'results/eval_diversity_5_1/Qwen_Qwen2.5-3B-Instruct.jsonl',
    'results/eval_diversity_5_1/Qwen_Qwen2.5-32B-Instruct.jsonl',
    'results/eval_prompt3_30gen_30gen_30gen_30gen_80/Qwen_Qwen2.5-3B-Instruct.jsonl'
]

humaneval_data = {'baseline1' : [], 'baseline2' : [], 'ours' : []}

for path_id, path in enumerate(result_paths):
    data = open_json(path)
    for item in data:
        if 'nodes' in item:
            for i in range(0, path.count('gen')):
                nodes = [node for node in item['nodes'] if len(node['parent_id']) == i]
                if i == 0:
                    initial_node = nodes[0]['content'] 
                node_contents = [node['content'] for node in nodes]
                if i == path.count('gen') - 1:
                    humaneval_data['ours'].append({'initial': initial_node, 'result': random.sample(node_contents, min(len(node_contents), 10))})
        else:
            initial_node = item['prompt'].split('Starting from the position of ')[-1].split(', please provide 10')[0]
            result_splits = item['result'].split("\n")
            if len([r for r in result_splits if 'Step5:' in r]) == 0:
                paths = [r.split('→ ')[-1].strip() for r in item['result'].split("\n") if 'Path' in r]
            else:
                paths = [r.split('Step5: ')[-1].strip() for r in result_splits if 'Step5:' in r]
            humaneval_data['baseline1' if path_id == 0 else 'baseline2'].append({'initial': initial_node, 'result': random.sample(paths, min(len(paths), 10))})

real_human_eval = []
for item in humaneval_data['ours']:
    initial = item['initial']
    ours_results = item['result']
    baseline1_results = [item['result'] for item in humaneval_data['baseline1'] if item['initial'] == initial][0]
    baseline2_results = [item['result'] for item in humaneval_data['baseline2'] if item['initial'] == initial][0]
    real_human_eval.append({'initial': initial, 'ours': '\n'.join(ours_results), 'baseline1': '\n'.join(baseline1_results), 'baseline2': '\n'.join(baseline2_results)})
# Soundedness: The job recommendation should be sounded and reasonable.

df = pd.DataFrame(real_human_eval)
data = [{**r, 'inputs': [r['initial'], r['ours'], r['baseline1'], r['baseline2']]} for r in real_human_eval]
prompt = """These are 3 kinds of Job recommendation for given path. Please select the most appropriate one.
- Please answer the option number only.

[Appropriate Criteria]:
Diversity: The job recommendation should be diverse and cover a wide range of job types.


[Starting path]: {}
[Option 1]:
{}
[Option 2]:
{}
[Option 3]:
{}
"""

results = get_results(
    model_name_or_path="claude-3-5-sonnet-20241022",
    prompt=prompt,
    data=data,
    max_tokens=50,
    batch_size=1,
    save_path='results/gpteval.json',
)

result = [extract_num(r[:10]) for r in results]
print(result)
print(Counter(result))