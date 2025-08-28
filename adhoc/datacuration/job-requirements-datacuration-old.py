import json

with open('data\linkedin_requirements_parsing_0.json','r',encoding='utf-8') as f:
    json_data = json.load(f)

prompt_list = []
for data in json_data:
    input = f"Answer the requirements for the given job.\n{data['job']}"
    output = ''
    for skills in data['requirements']:
        output += f'# {skills}\n'
        for skill in data['requirements'][skills]:
            output += f'## {skill}\n'
            for sub_skill in data['requirements'][skills][skill]:
                output += f'* {sub_skill}\n'
        output += '\n'
    prompt_list.append({'messages': [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': output}]})

import random

random.seed(0)
random.shuffle(prompt_list)

training_data = prompt_list[:-100]
test_data = prompt_list[-100:]

with open('data\linkedin_requirements_parsing_training_data.jsonl','w',encoding='utf-8') as f:
    for d in training_data:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data\linkedin_requirements_parsing_test_data.jsonl','w',encoding='utf-8') as f:
    for d in test_data:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')
