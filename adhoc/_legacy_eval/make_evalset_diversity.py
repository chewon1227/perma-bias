from mylmeval import open_json, save_json
import os
import pandas as pd
import re
from tqdm import tqdm, trange
import networkx as nx
import matplotlib.pyplot as plt
import fire
from typing import List, Dict, Optional

group1 = [(22, 'LinkedIn people profiles datasets.csv')]
group2 = [(3, '10k_data_li_japan.txt'), (4, '10000_random_israel_people_profiles.txt'), (5, '10k_data_li_india.txt'), (15, 'singapore_profiles.txt'), (19, 'danish_people.txt'), (20, '10000_random_us_people_profiles.txt'), (21, '10000_random_canadian_profiles.txt')]
group3 = [(16, 'Survey on Employment Trends  2.csv')]

DEGREE_ABBR = {
   "Btech": "Bachelor of Technology",
   'Diploma ' : 'Diploma',
   'diploma' : 'Diploma',
   "BE": "Bachelor of Engineering",
   "MTech": "Master of Technology",
   "M.Tech": "Master of Technology", 
   "BCA": "Bachelor of Computer Applications",
   "MCA": "Master of Computer Applications",
   "BSc": "Bachelor of Science",
   "MSc": "Master of Science",
   "BCom": "Bachelor of Commerce",
   "BBA": "Bachelor of Business Administration",
   "MBA": "Master of Business Administration",
   "MSW": "Master of Social Work",
   "BDS": "Bachelor of Dental Surgery",
   "Pharm D": "Doctor of Pharmacy",
   "BVOC": "Bachelor of Vocation",
   "M Eng": "Master of Engineering",
   "MA": "Master of Arts",
   "ITI": "Industrial Training Institute",
   "EEE": "Electrical and Electronics Engineering"
}

def open_file(data_path, original_dir='data/linkedin'):
    if data_path.endswith('.csv'):
        data = pd.read_csv(f"{original_dir}/{data_path}", encoding='utf-8', encoding_errors='replace')
    elif data_path.endswith('.xlsx'):
        data = pd.read_excel(f"{original_dir}/{data_path}")
    elif data_path.endswith('.txt'): # convert to jsonl
        data = pd.read_json(f"{original_dir}/{data_path}", lines=True)
    return data.to_dict(orient='records')


def get_date_key(x):
    starts_at = x['starts_at']
    if starts_at == None:
        return 0
    try:
        return int(re.sub(r'[^0-9]', '', starts_at))
    except:
        return int(starts_at['year'])

def sort_track(track: List[Optional[Dict]]) -> List[Dict]:
    valid_track = [r for r in track if r is not None]
    if not valid_track:
        return []
        
    df = pd.DataFrame(valid_track)
    
    # content 데이터 추출 및 새로운 컬럼 생성
    content_data = []
    for content in df['content']:
        content_data.append({
            'main': content['main'],
            'detail': content.get('detail', None)
        })
    
    # 새로운 데이터프레임 생성 및 컬럼 추가
    content_df = pd.DataFrame(content_data)
    df['main'] = content_df['main']
    df['detail'] = content_df['detail']
    
    # 필요한 컬럼만 선택
    df = df.drop_duplicates(subset=['main', 'detail'], keep='first')
    columns_to_keep = [r for r in df.columns if r not in ['main', 'detail']]
    df = df[columns_to_keep]
    
    result = df.to_dict(orient='records')
    
    return sorted(result, key=get_date_key)

def p_exp(xi):
    re = []
    if isinstance(xi, str):
        for x in eval(xi.replace('null', 'None').replace('NoneType', 'None')):
            if 'title' in x:
                re.append({
                    'content' : {'main' : x['title'], 'detail' : x['location']} if 'location' in x else {'main' : x['title']},
                    'starts_at' : x['start_date'] if 'start_date' in x else '0',
                    'ends_at' : x['end_date'] if 'end_date' in x else None,
                })
            elif 'positions' in x:
                try:
                    re.append({
                        'content' : {'main' : x['positions'][0]['title'], 'detail' : x['company']} if 'company' in x else {'main' : x['positions'][0]['title']},
                        'starts_at' : x['positions'][0]['start_date'] if 'start_date' in x['positions'][0] else '0',
                        'ends_at' : x['positions'][0]['end_date'] if 'end_date' in x['positions'][0] else None,
                    })
                except:
                    print('error')
            else:
                import pdb; pdb.set_trace()
    return re

def p_edu(xi):
    re = []
    if isinstance(xi, str):
        for x in eval(xi):
            try:
                end_year = int(x['end_year'])
            except:
                end_year = None
            re.append({
                'content' : {'main' : x['degree'], 'detail' : x['title']} if 'degree' in x else {'main' : x['title']},
                'starts_at' : x['start_year'] if 'start_year' in x and x['start_year'] != '' else '0',
                'ends_at' : end_year,
            })
    return re


def to_diversity_file():
    result = []
    current_idx = 0
    
    # Group 1 처리
    for group in group1:
        exp_key = 'experience'
        edu_key = 'education'
        job_key = 'position'
        
        data = open_file(group[1])
        
        for d in tqdm(data):
            track = sort_track(p_edu(d[edu_key])) + sort_track(p_exp(d[exp_key]))
            valid_track = []
            
            # 먼저 유효한 항목만 필터링
            for id, t in enumerate(track):
                if t['content'].get('main') not in [None, '', 'None']:
                    valid_track.append((id, t))
            
            # 유효한 항목이 있는 경우에만 처리
            if valid_track:
                for node_idx, (original_id, t) in enumerate(valid_track):
                    result.append({
                        'idx': current_idx,
                        'node': node_idx,  # 연속적인 node 번호 사용
                        'from': None if node_idx == 0 else node_idx-1,
                        'to': None if node_idx == len(valid_track)-1 else node_idx+1,
                        'type': 'step',
                        'content': t['content'],
                        'meta': {'source': group[1], 'lang': d["country_code"].lower() if isinstance(d["country_code"], str) else 'en'}
                    })
                current_idx += 1  # 유효한 track 하나를 완전히 처리한 후에 증가

    # Group 3 처리    
    for group in group3:    
        req_key = 'Skills that you are confident '
        edu_key = 'Educational Qualification '
        job_key = 'Interested area of work'
        
        def edu(xi):
            def _process_degree(x):
                for k, v in DEGREE_ABBR.items():
                    x = x.replace(k, v)
                return x
            new_xi = _process_degree(xi.strip(" "))
            return new_xi if xi != None else None
        
        data = open_file(group[1])
        
        for r in data:
            valid_records = []
            edu_content = edu(r[edu_key])
            job_content = r[job_key]
            
            if edu_content and edu_content not in [None, '', 'None']:
                valid_records.append({
                    'idx': current_idx,
                    'node': 0,
                    'from': None,
                    'to': 1,
                    'type': 'step',
                    'content': {'main': edu_content},
                    'meta': {'source': group[1], 'lang': 'en'}
                })
            
            if job_content and job_content not in [None, '', 'None']:
                valid_records.append({
                    'idx': current_idx,
                    'node': 1,
                    'from': 0,
                    'to': None,
                    'type': 'step',
                    'content': {'main': job_content},
                    'meta': {'source': group[1], 'lang': 'en'}
                })
                
            for id, req in enumerate(r[req_key].split(";")):
                if req.strip() not in [None, '', 'None']:
                    valid_records.append({
                        'idx': current_idx,
                        'node': id+2,
                        'from': 0,
                        'to': 1,
                        'type': 'requirement',
                        'content': {'main': req.strip()},
                        'meta': {'source': group[1], 'lang': 'en'}
                    })
            
            if valid_records:
                result.extend(valid_records)
                current_idx += 1
        
    # Group 2 처리
    for group in group2:
        exp_key = 'experiences'
        edu_key = 'education'
        job_key = 'occupation'
        
        def process_exp(x):
            if 'title' in x and x['title'] != None and 'none' not in x['title'].lower():
                if 'company' in x and x['company'] != None and 'none' not in x['company'].lower():
                    content = {'main': x['title'], 'detail': x['company']}
                else:
                    content = {'main': x['title']}
            elif 'company' in x and x['company'] != None and 'none' not in x['company'].lower():
                content = {'main': x['company']}
            else:
                return None                    
            return {'content': content, **{k:v for k,v in x.items() if k in ['starts_at', 'ends_at']}}
        
        def process_edu(x):
            priority = ['degree_name', 'field_of_study']
            main = ''
            detail = None
            for p in priority:
                if p in x and x[p] != None and x[p].lower() != 'none':
                    main += x[p] if main == '' else f", {x[p]}"
            if 'school' in x and x['school'] != None and x['school'].lower() != 'none':
                if main == '':
                    main = x['school']
                    content = {'main': main}
                else:
                    detail = x['school']     
                    content = {'main': main, 'detail': detail}              
                return {'content': content, **{k:v for k,v in x.items() if k in ['starts_at', 'ends_at']}}

        def process_country(x):
            if x == 'jp':
                return 'ja'
            elif x == 'kr':
                return 'ko'
            elif x in ['us', 'uk', 'ca', 'au', 'nz', 'ie']:
                return 'en'
            else:
                return x

        data = open_file(group[1])
        for d in tqdm(data):
            edus = [process_edu(r) for r in d[edu_key]]
            exps = [process_exp(r) for r in d[exp_key]]
            track = sort_track(edus) + sort_track(exps)
            
            valid_records = []
            for id, exp in enumerate(track):
                if exp['content'].get('main') not in [None, '', 'None']:
                    valid_records.append({
                        'idx': current_idx,
                        'node': id,
                        'from': None if id == 0 else id-1,
                        'to': None if id == len(track)-1 else id+1,
                        'type': 'step',
                        'content': exp['content'],
                        'meta': {'source': group[1], 'lang': process_country(d['country'].lower())}
                    })
            
            if valid_records:
                result.extend(valid_records)
                current_idx += 1

    print(f"Total records created: {len(result)}")
    print(f"Total unique indices: {current_idx}")
    
    print("Renumbering nodes within each graph...")
    result = renumber_nodes(result)
    
    # None 값 처리 및 저장
    final_result = []
    for r in result:
        if r['content'].get('main') in [None, '', 'None'] and r['content'].get('detail'):
            r['content']['main'] = r['content']['detail']
            r['content'].pop('detail', None)
        final_result.append(r)
    
    print(f"Saving {len(final_result)} records...")
    save_json(final_result, 'data/evalset/diversity.jsonl')
    print("Data saved successfully")
    
def renumber_nodes(records):
    """
    각 그래프 내의 노드들을 연속적인 번호로 리넘버링하고,
    step 타입의 노드들이 하나의 체인으로 이어지도록 보장합니다.
    """
    graphs = {}
    for record in records:
        idx = record['idx']
        if idx not in graphs:
            graphs[idx] = []
        graphs[idx].append(record)
    
    new_records = []
    for idx, graph_records in graphs.items():
        # step 노드와 다른 타입의 노드 분리
        step_nodes = [r for r in graph_records if r['type'] == 'step']
        other_nodes = [r for r in graph_records if r['type'] != 'step']
        
        if not step_nodes:  # step 노드가 없는 경우
            continue
            
        # step 노드들의 순서 결정
        step_chain = []
        remaining_steps = step_nodes.copy()
        
        # 시작 노드 찾기 (from이 None인 노드)
        start_nodes = [n for n in remaining_steps if n['from'] is None]
        if not start_nodes:  # 시작 노드가 없으면 임의의 노드를 시작점으로
            start_nodes = [remaining_steps[0]]
        
        # 하나의 시작점만 유지
        current_node = start_nodes[0]
        step_chain.append(current_node)
        remaining_steps.remove(current_node)
        
        # 나머지 노드들을 순서대로 연결
        while remaining_steps:
            next_nodes = [n for n in remaining_steps if n['from'] == current_node['node']]
            if not next_nodes:  # 다음 노드를 찾을 수 없으면 남은 노드 중 하나를 선택
                next_nodes = [remaining_steps[0]]
            
            current_node = next_nodes[0]
            step_chain.append(current_node)
            remaining_steps.remove(current_node)
        
        # 노드 번호 재할당
        old_to_new = {}
        
        # step 노드들 먼저 번호 할당
        for new_idx, node in enumerate(step_chain):
            old_to_new[node['node']] = new_idx
        
        # 다른 타입의 노드들에 대한 번호 할당
        other_start_idx = len(step_chain)
        for new_idx, node in enumerate(other_nodes, start=other_start_idx):
            old_to_new[node['node']] = new_idx
        
        # 새로운 레코드 생성
        for i, node in enumerate(step_chain):
            new_record = node.copy()
            new_record['node'] = old_to_new[node['node']]
            new_record['from'] = None if i == 0 else i - 1
            new_record['to'] = None if i == len(step_chain) - 1 else i + 1
            new_records.append(new_record)
        
        # 다른 타입의 노드들 처리
        for node in other_nodes:
            new_record = node.copy()
            new_record['node'] = old_to_new[node['node']]
            # from과 to는 원래 값 유지하되 새로운 번호로 매핑
            if node['from'] is not None:
                new_record['from'] = old_to_new.get(node['from'], None)
            if node['to'] is not None:
                new_record['to'] = old_to_new.get(node['to'], None)
            new_records.append(new_record)
    
    return new_records


if __name__ == '__main__':
    fire.Fire(to_diversity_file)