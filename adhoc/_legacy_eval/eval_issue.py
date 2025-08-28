from mylmeval import save_json, open_json, get_results
import fire
import json
import pandas as pd
from collections import defaultdict
import os
import random
from careerpathway.scoring import load_issue_pathways, get_prompt_and_model
import re
import numpy as np
import collections
from collections import Counter
from typing import List, Dict



def _make_options(options):
    return '\n'.join([f"{i+1}. {r}" for i, r in enumerate(options)])

def _extract_num(text):
    # if number in text
    try:
        return int(re.findall(r'\d+', text)[0])
    except:
        return None

def int_to_capital(num):
    """Convert integer to capital letter. 1 -> A, 2 -> B, ..."""
    return chr(num + 64)

def make_v1_v5(value_data):
    # 먼저 v1-v5 값들을 추출
    value_data = [{
        **r,
        'v1': r['result'].split("\n")[0].split("1. ")[-1].strip() if len(r['result'].split("\n")) == 5 else None,
        'v2': r['result'].split("\n")[1].split("2. ")[-1].strip() if len(r['result'].split("\n")) == 5 else None,
        'v3': r['result'].split("\n")[2].split("3. ")[-1].strip() if len(r['result'].split("\n")) == 5 else None,
        'v4': r['result'].split("\n")[3].split("4. ")[-1].strip() if len(r['result'].split("\n")) == 5 else None,
        'v5': r['result'].split("\n")[4].split("5. ")[-1].strip() if len(r['result'].split("\n")) == 5 else None,
    } for r in value_data]
    
    # superior와 superior_cnt 계산
    processed_data = []
    for r in value_data:
        # None이 하나라도 있는지 확인
        values = [r[f'v{i}'] for i in range(1,6)]
        if None in values:
            r.update({
                'superior': None,
                'superior_cnt': None
            })
        else:
            # Option A와 Option B의 개수를 세기
            count = Counter(values)
            option_a_count = count.get('Option A', 0)
            option_b_count = count.get('Option B', 0)
            
            # superior 결정
            if option_a_count > option_b_count:
                superior = 'Option A'
                superior_cnt = option_a_count
            else:
                superior = 'Option B'
                superior_cnt = option_b_count
                
            r.update({
                'superior': superior,
                'superior_cnt': superior_cnt
            })
            
        processed_data.append(r)
    
    return processed_data


def score_with_gt(data, value_data):
    values = defaultdict(list)
    cnt = 0
    parsed_result = []
    for item in data:
        if '1' in item['result'] or '2' in item['result']:
            answer = _extract_num(item['result'].split("Answer")[-1])
            if answer == None:
                answer = _extract_num(item['result'])
            parsed_result.append({**item, 'answer' : answer})
                
        else:
            parsed_result.append({**item, 'answer' : None})
    # print(sum([value['superior'] == parsed['answer'] for value, parsed in zip(value_data, parsed_result)]))
    for value, item in zip(value_data, parsed_result):
        if item['answer'] in [1,2]:
            option = 'Option '+int_to_capital(item['answer'])
            if option == value['superior']:
                cnt += 1
            for i in range(1,6):
                if value[f"v{i}"] and option in value[f'v{i}']:
                    values[f'v{i}'].append(item)
    return cnt, values
        
        
def main(
    make_score: bool = False,
    model_name_or_path: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    start: int = None,
    ):

    prompt = {
    'ko' :   """당신의 할 일은 어떤 사람의 진로에 대한 고민을 보고, 그가 선택할 수 있는 여러 선택지 중 가장 적절하고 추천할 만한 것을 하나 고르는 것입니다.
- 제시된 text를 보고 choices를 충분히 고려합니다.
- 어떠한 이유로 어떤 선택지가 가장 적절한지 '[Explain]'으로 시작하는 이유를 설명하시오.
- 마지막에 '[Answer]'로 시작하고 번호와 답을 제시하시오.

[제목]: {}
[Text]: {}
[선택지]:
{}""",

    'en' : """Your task is to look at someone's concerns about their career path and choose the most appropriate and recommended option among the various choices they can make.
- Read the given text and carefully consider the choices.
- Explain why which option is the most appropriate by starting with '[Explain]'.
- Begin with '[Answer]' at the end and provide a number and answer.

[Title]: {}  
[Text]: {}

[Options]:
{}"""
    }
    save_dir = 'results/eval_issue_pathway'
    os.makedirs(f'{save_dir}', exist_ok=True)
    if make_score:
        # salary_data = open_json('data/evalset/issue_pathway_salary.jsonl')
        value_data = open_json('data/evalset/issue_pathway_values_gpt-4o.jsonl')
        value_data = make_v1_v5(value_data)
        print(Counter([r['superior_cnt'] for r in value_data]))
        print([[r['v1'], r['v2'], r['v3'], r['v4'], r['v5']] for r in value_data if r['superior_cnt'] == 1])
        print([[r['v1'], r['v2'], r['v3'], r['v4'], r['v5']] for r in value_data if r['superior_cnt'] == 0])
        
        for file_name in os.listdir(save_dir):
            data = open_json(f'{save_dir}/{file_name}')
            concurrence, values = score_with_gt(data, value_data)
            print(file_name)
            print(concurrence)
            print([len(values[f'v{i}']) for i in range(1,6)])
        return  
    
    os.makedirs(f'{save_dir}', exist_ok=True)
    
    testset, _ = load_issue_pathways()
    options = open_json('/home/iyy1112/workspace/Career-Pathway/data/issue_options.jsonl')
    data = [
        {'inputs' : [
            prompt[item['language']].format(item['title'], item['issue'], _make_options([options[id]['option1'], options[id]['option2']]))
            ]}
        for id, item in enumerate(testset)
    ]
    
    print(data[0]['inputs'][0])
    
    results = get_results(
        model_name_or_path=model_name_or_path,
        data=data[start:] if start != None else data,
        prompt='{}',
        max_tokens=1024,
        batch_size=len(data),
        apply_chat_template='auto',
        save_path=f'{save_dir}/{model_name_or_path.replace("/", "_")}{f"_{start}" if start != None else ""}.jsonl'
    )


def annotate_data(
    type : str = 'salary', # requirements
    start: int = None,
    model_name_or_path: str = 'gpt-4o',
    ):
    
    data, _ = load_issue_pathways()
    options = open_json('data/issue_options.jsonl')
    print(len(data), len(options))

    if type in ['salary', 'requirements']:
        prompt, model_name_or_path = get_prompt_and_model(type)
        salary_prompt = {
            'en' : "Answer the average annual salary for the following job.\n{}",
            'ko' : "다음 직업의 평균 연봉을 대답하세요.\n{}"
        }
        data = [
        {'inputs' : [
            prompt[item['language']].format(option[f'option{i}'])
            ]}
        for item, option in zip(data, options) for i in range(1, 3)
        ]
    elif type == 'values':
        prompt = {
            'ko' : """다음은 진로 고민과 그에 대한 선택지 2(Option A, OptionB)이다. 아래에는 선택을 할 때 고려해야 할 다섯 가지 요소들이며, 각 요소에 대하여 더 나은 선택지를 고르시오.
- 다섯 가지 선택지에 대하여 각각 Option 1과 Option 2 중 어느 것이 더 적합한지 고르시오.
- 선택지를 제시할 때는 숫자를 붙이고, 선택지 외의 다른 텍스트는 부가하지 않는다. (예: 1. Option A, 2. Option B)

[요소]
1. 적성 부합도 : 자신의 강점과 약점이 해당 진로와 얼마나 잘 맞는지, 실제로 그 일을 잘 해낼 수 있는 능력이 있는지, 장기적으로 전문성을 키워나갈 수 있는지
2. 흥미/열정 수준 : 그 일을 하면서 지속적인 동기부여가 될 수 있는지, 일 자체에서 의미와 보람을 찾을 수 있는지, 힘들 때도 계속할 수 있는 내적 동기가 있는지
3. 시장 수요와 전망 : 현재와 미래의 일자리 시장에서 얼마나 필요로 하는 직종인지, 산업 트렌드와 기술 발전을 고려했을 때 지속가능한지, 향후 성장 가능성이 있는지
4. 경제적 보상 : 투자하는 시간과 노력 대비 적절한 보상이 있는지, 원하는 삶의 수준을 유지할 수 있는 수입이 보장되는지, 장기적인 재정적 안정성이 있는지
5. 실현 가능성 : 현실적으로 달성 가능한 목표인지, 필요한 자격이나 조건을 갖출 수 있는지, 진입 장벽을 극복할 수 있는지

[고민]: {}. {}
[Option A]: {}
[Option B]: {}
""",
            'en' : """The following is a career concern and two options (Option A, Option B) for it. Below are five factors to consider when making a choice, and choose the better option for each factor.
- Choose which of Option 1 and Option 2 is more appropriate for each of the five factors.
- Number the options and do not add any additional text other than the options. (e.g. 1. Option A, 2. Option B)

[Factors]
1. Suitability : How well your strengths and weaknesses match the career, whether you have the ability to do the job well, and whether you can develop expertise in the long run
2. Interest/Passion : Whether you can be continuously motivated by doing the job, whether you can find meaning and reward in the job itself, and whether you have internal motivation to continue even when it is difficult
3. Market Demand and Prospects : How much the job is needed in the current and future job market, whether it is sustainable considering industry trends and technological developments, and whether there is growth potential in the future
4. Economic Reward : Whether there is appropriate compensation for the time and effort invested, whether there is guaranteed income to maintain the desired standard of living, and whether there is long-term financial stability
5. Feasibility : Whether the goal is realistically achievable, whether you can meet the necessary qualifications or conditions, and whether you can overcome entry barriers

[Issue]: {}. {}
[Option A]: {}
[Option B]: {}
"""
        }
        data = [
            {
                'inputs' : [prompt[item['language']].format(item['title'], item['issue'], option['option1'], option['option2'])]
            }
            for item, option in zip(data, options)
        ]
    # data = [{'inputs' : [k], 'item_id': id, 'option_id' : iid} for id, r in enumerate(data) for iid, k in enumerate(r['options'])]    
    
    results = get_results(
        model_name_or_path=model_name_or_path,
        data=data[start:start+3000] if start != None else data,
        prompt="{}",
        max_tokens=1024,
        batch_size=len(data),
        apply_chat_template='auto',
        save_path=f'data/evalset/issue_pathway_{type}_{model_name_or_path}_{start}.jsonl'
    )



def datacuration(
    model_name_or_path: str = 'Qwen/Qwen2.5-32B-Instruct',
    start: int = None,
):
    save_dir = 'data'
    
    prompt = {
        'ko' : """다음은 진로 고민에 대한 내용이다. 이 사람에게 추천해줄 수 있는 다음 선택 경로로, 가장 가능성이 있으며 적합한 것을 2가지 제시하여라.
- 각 선택지는 한 줄로만 작성하며 선택지 외의 다른 텍스트는 부가하지 않는다.
- 두 선택지는 각각 다른 방향으로 나아가는 것으로 해야 한다.
- 선택지를 제시할 때는 숫자를 붙여라.

[제목]: {}
[Text]: {}
[선택지]:""",
        'en' : """The following is about career concerns. Provide two possible and appropriate next career steps for this person.
- Write each option in one line without any additional text.
- The two options should be in different directions.
- Number the options.

[Title]: {}
[Text]: {}
[Options]:"""
    }
    testset, _ = load_issue_pathways()
    data = [
        {'inputs' : [
            prompt[item['language']].format(item['title'], item['issue'])
            ]}
        for item in testset
    ]
    
    print(data[0]['inputs'][0])
    
    results = get_results(
        model_name_or_path=model_name_or_path,
        data=data[start:] if start != None else data,
        prompt='{}',
        max_tokens=1024,
        batch_size=len(data),
        apply_chat_template='auto',
        save_path=f'{save_dir}/{f"dilemma_{start}" if start != None else "dilemma"}.jsonl'
    )
    
    
if __name__ == "__main__":
    fire.Fire({
        'main' : main,
        'annotate_data' : annotate_data,
        'datacuration' : datacuration
    })