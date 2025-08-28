import json
import pandas as pd

def _remove_str_in_paren(s):
    """괄호 안의 문자열을 제거하고 앞뒤 문자열을 합침"""
    if '(' not in s or ')' not in s:
        return s.strip()
    saved1 = s.split('(')[0].strip()
    saved2 = s.split(')')[-1].strip()
    new = ' '.join([saved1, saved2]).strip()
    
    saved1 = new.split('[', 1)[0].strip()
    saved2 = new.split(']', 1)[-1].strip()
    return ' '.join([saved1, saved2]).strip()
    
def get_skill_data(data, types):
    """스킬 데이터 생성"""
    inst_data = []
    for item in data:
        if not item.get('skills'):
            continue
            
        skill_dict = {'Apt': [], 'Tech': []}
        for skill in item['skills']:
            if skill.lower() in types:
                skill_dict[types[skill.lower()]].append(skill)
                
        inst_data.append({
            'job': _remove_str_in_paren(item['job']).capitalize(),
            'postname': _remove_str_in_paren(item['postname']).capitalize(),
            'output': skill_dict
        })
    
    return inst_data

def process_duplicates(df):
    """중복된 postname을 기준으로 데이터 처리"""
    # 중복된 postname만 필터링
    duplicated_posts = df.duplicated(subset=['postname'], keep=False)
    filtered_df = df[duplicated_posts]
    
    # 그룹별로 output을 합치고 중복 제거
    def combine_outputs(x):
        result = {'Apt': [], 'Tech': []}
        for output in x:
            result['Apt'].extend(output['Apt'])
            result['Tech'].extend(output['Tech'])
        result['Apt'] = list(set(result['Apt']))
        result['Tech'] = list(set(result['Tech']))
        return result
    
    result = filtered_df.groupby(['postname']).agg({
        'job': 'first',
        'output': combine_outputs
    }).reset_index()
    
    return result

with open('data/data13_15_kaggle.jsonl', 'r', encoding='utf-8') as f:
    data = f.readlines()
data = [json.loads(item) for item in data]

with open('data/linkedin_requirements_parsing7.json', 'r', encoding='utf-8') as f:
    type_data = json.load(f)

apt_dict = {}
tech_dict = {}
for item in type_data:
    if 'requirements' in item:
        if 'Aptitude-related Skills' in item['requirements']:
            for category, apt_skill_list in item['requirements']['Aptitude-related Skills'].items():
                if isinstance(apt_skill_list, list):
                    for skill in apt_skill_list:
                        if skill.lower() not in apt_dict:
                            apt_dict[skill.lower()] = 0
                        apt_dict[skill.lower()] += 1
        if 'Technical Skills' in item['requirements']:
            for category, tech_skill_list in item['requirements']['Technical Skills'].items():
                if isinstance(tech_skill_list, list):
                    for skill in tech_skill_list:
                        if skill.lower() not in tech_dict:
                            tech_dict[skill.lower()] = 0
                        tech_dict[skill.lower()] += 1

types = {}
for skill in apt_dict:
    if skill in tech_dict:
        if apt_dict[skill] > tech_dict[skill]:
            types[skill] = 'Apt'
        else:
            types[skill] = 'Tech'
    else:
        types[skill] = 'Apt'

for skill in tech_dict:
    if skill not in types:
        types[skill] = 'Tech'

skill_data = get_skill_data(data, types)
skill_df = pd.DataFrame(skill_data)
skill_result = process_duplicates(skill_df)
skill_list = skill_result.to_dict(orient='records')

with open('data/data19_en_onet.jsonl', 'r', encoding='utf-8') as f:
    onet_data = f.readlines()

onet_data = [json.loads(item) for item in onet_data]
onet_jobname = [item['job'] for item in onet_data]

prompt_list = []
for d in skill_list:
    job = d['job']
    for onet_job in onet_jobname:
        if onet_job.lower() in d['postname'].lower():
            job = d['postname']
            print(job)
            break
    input = f"Answer the requirements for the given job.\n{job}"
    output = f"[Aptitude] {', '.join(d['output']['Apt'])}\n[Technical] {', '.join(d['output']['Tech'])}"
    prompt_list.append({'messages': [{'role': 'user', 'content': input}, {'role': 'assistant', 'content': output}]})

import random

random.seed(0)
random.shuffle(prompt_list)

training_data = prompt_list[:-700]
test_data = prompt_list[-700:]

with open('data\linkedin_requirements_training_data.jsonl','w',encoding='utf-8') as f:
    for d in training_data:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data\linkedin_requirements_test_data.jsonl','w',encoding='utf-8') as f:
    for d in test_data:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')
