import json

WAGE_INCREASE_RATE = 1.112809538367472 # 3640 / 3271

data=[]
with open('data/evalset/truthfulness.jsonl','r',encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# delete except task=='salary'
data=[d for d in data if d['task']=='salary']

# increase wage if language=='ko' and round down to the nearest 10000
for d in data:
    if d['language']=='ko':
        d['groundtruth']=int(d['groundtruth']*WAGE_INCREASE_RATE // 10000 * 10000)

new_data=[]
for d in data:
    new_data.append({
        'input':d['input'],
        'language':d['language'],
        'salary':d['groundtruth'],
    })

with open('data/evalset/job-salary-data.json','w',encoding='utf-8') as f:
    json.dump(new_data,f,indent=2,ensure_ascii=False)
    
dataset = []

for d in new_data:
    unit = 'USD'
    prompt = 'Answer the average annual salary for the following job.'
    if d['language'] == 'ko':
        unit = 'KRW'
        prompt = '다음 직업의 평균 연봉을 대답하세요.'
    elif d['language'] == 'ja':
        unit = 'JPY'
        prompt = '次の職業の平均年収を答えてください。'
    elif d['language'] == 'es':
        unit = 'EUR'
        prompt = 'Responda al salario anual promedio para el siguiente trabajo.'
    message = {'messages': [{'role': 'user', 'content': f'{prompt}\n{d["input"]}'},
                            {'role': 'assistant', 'content': f'{unit} {d["salary"]}'}]}
    dataset.append(message)

import random
random.seed(0)
random.shuffle(dataset)

test_dataset = dataset[-100:]
training_dataset = dataset[:-100]

with open('data/evalset/job-salary-training-data.jsonl','w',encoding='utf-8') as f:
    for d in training_dataset:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data/evalset/job-salary-test-data.jsonl','w',encoding='utf-8') as f:
    for d in test_dataset:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')