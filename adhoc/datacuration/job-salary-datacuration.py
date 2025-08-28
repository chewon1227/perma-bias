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
    prompt = 'Answer the salary for the following job.'
    if d['language'] == 'ko':
        unit = 'KRW'
        prompt = '다음 직업의 급여를 대답하세요.'
    elif d['language'] == 'ja':
        unit = 'JPY'
        prompt = '次の職業の給与を答えてください。'
    elif d['language'] == 'es':
        unit = 'EUR'
        prompt = 'Responde al salario del siguiente puesto de trabajo.'
    message = {'messages': [{'role': 'user', 'content': f'{prompt}\n{d["input"]}'},
                            {'role': 'assistant', 'content': f'{unit} {d["salary"]}'}]}
    dataset.append(message)

dataset_with_per = []
for d in dataset:
    peryear = f'{d["messages"][1]["content"]} per year'
    dataset_with_per.append({'messages': [{'role': 'user', 'content': d['messages'][0]['content']},
                                          {'role': 'assistant', 'content': peryear}]})

with open('data/salary_inst_data.json', 'r', encoding='utf-8') as f:
    salary_inst_data = json.load(f)

def convert_to_hour(salary, per):
    if per == 'year':
        return salary / 52 / 40
    elif per == 'month':
        return salary / 4 / 40
    elif per == 'week':
        return salary / 40
    elif per == 'day':
        return salary / 8
    elif per == 'hour':
        return salary

def convert_to_year(salary, per):
    if per == 'year':
        return salary
    elif per == 'month':
        return salary * 12
    elif per == 'week':
        return salary * 52
    elif per == 'day':
        return salary * 5 * 52
    elif per == 'hour':
        return salary * 8 * 5 * 52

salary_data = []
salary_data_with_per = []
for d in salary_inst_data:
    if d['output'][0]['type'] == 'medium':
        medium_salary = d['output'][0]['value']
        MEDIUM = True
    else:
        min_salary = d['output'][0]['value']
        max_salary = d['output'][1]['value']
        MEDIUM = False
    per = d['output'][0]['per']
    unit = d['output'][0]['unit']
    if MEDIUM:
        if convert_to_hour(medium_salary, per) <= 5:
            continue
        if convert_to_year(medium_salary, per) > 1000000:
            continue
    else:
        if convert_to_hour(min_salary, per) <= 5:
            continue
        if convert_to_year(min_salary, per) > 1000000:
            continue
        if convert_to_year(max_salary, per) > 1000000:
            continue
    prompt = 'Answer the salary for the following job.'
    if MEDIUM:
        message = {'messages': [{'role': 'user', 'content': f'{prompt}\n{d["input"]}'},
                                {'role': 'assistant', 'content': f'{unit} {convert_to_year(medium_salary, per)}'}]}
        message_with_per = {'messages': [{'role': 'user', 'content': f'{prompt}\n{d["input"]}'},
                                {'role': 'assistant', 'content': f'{unit} {medium_salary} per {per}'}]}
    else:
        message = {'messages': [{'role': 'user', 'content': f'{prompt}\n{d["input"]}'},
                                {'role': 'assistant', 'content': f'{unit} min {convert_to_year(min_salary, per)} - max {convert_to_year(max_salary, per)}'}]}
        message_with_per = {'messages': [{'role': 'user', 'content': f'{prompt}\n{d["input"]}'},
                                {'role': 'assistant', 'content': f'{unit} min {min_salary} - max {max_salary} per {per}'}]}
    salary_data.append(message)
    salary_data_with_per.append(message_with_per)

salary_data.extend(dataset)
salary_data_with_per.extend(dataset_with_per)

import random

random.seed(0)
random.shuffle(salary_data)
random.shuffle(salary_data_with_per)

salary_data_training = salary_data[:-2000]
salary_data_test = salary_data[-2000:]

salary_data_with_per_training = salary_data_with_per[:-2000]
salary_data_with_per_test = salary_data_with_per[-2000:]

with open('data/salary-training-data.jsonl','w',encoding='utf-8') as f:
    for d in salary_data_training:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data/salary-test-data.jsonl','w',encoding='utf-8') as f:
    for d in salary_data_test:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data/salary-training-data-with-per.jsonl','w',encoding='utf-8') as f:
    for d in salary_data_with_per_training:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data/salary-test-data-with-per.jsonl','w',encoding='utf-8') as f:
    for d in salary_data_with_per_test:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')
