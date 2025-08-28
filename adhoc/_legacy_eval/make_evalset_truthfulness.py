from mylmeval import open_json, save_json
import os
import pandas as pd


SOURCES = [
    'data/data16_ko_essensial.json', # ko
    'data/data18_ja_salary.json', # ja
    'data/data19_en_onet.jsonl', # en
    'data/data11_onet.json', # ja
    'data/data7_educaweb.json',
    'data/data20_en_onet_mfm.jsonl',
    'data/data5_jobdict.json', # ko
]


result = []
for i, source in enumerate(SOURCES):
    data = open_json(SOURCES[i])
    if i == 0:
        for d in data:
            result.append({
                'input' : d['직업명'],
                'groundtruth' : 'KRW '+ str(int(d['임금'].split("\n")[1].split(" ")[1].replace('만원', '').replace(',', '')) * 10000),
                'source': source,
                'language' : 'ko',
                'task' : 'salary',
                'else' : {'unit' : 'KRW', **{k:v for k, v in d.items() if k in ['업무수행능력']}}
            })
            result.append({
                'input' : d['직업명'],
                'groundtruth' : d['필요기술 및 지식'],
                'source': source,
                'language' : 'ko',
                'task' : 'requirement',
                'else' : {k:v for k, v in d.items() if k in ['업무수행능력']}
            })
            result.append({
                'input' : d['직업명'],
                'groundtruth' : d['하는 일'],
                'source': source,
                'language' : 'ko',
                'task' : 'description',
                'else' : {k:v for k, v in d.items() if k in ['업무수행능력']}
            })
            
    if i == 1:
        for d in data:
            result.append({
                'input' : d['job'],
                'groundtruth' : 'JPY '+ str(d['annual_salary']),
                'source': source,
                'language' : 'ja',
                'task' : 'salary',
                'else' : {'unit' : 'JPY', **{k:v for k, v in d.items() if k in ['job_category']}}
            })
            
    if i == 2:
        for d in data:
            if (d['salary'] != None) and (d['job'] != ''):
                result.append({
                    'input' : d['job'],
                    'groundtruth' : 'USD '+ str(d['salary'].split(" ")[-2].replace(',', '').replace('$', '')),
                    'source': source,
                    'language' : 'en',
                    'task' : 'salary',
                    'else' : {'unit' : 'USD', **{k:v for k, v in d.items() if k in ['job_description', 'company']}}
                })
        
            if 'Tasks' in d:
                result.append({
                    'input' : d['job'],
                    'groundtruth' : d['Tasks'],
                    'source': source,
                    'language' : 'en',
                    'task' : 'description',
                    'else' : {k:v for k, v in d.items() if k in ['job_description', 'company']}
                })
    
    if i == 3:
        for d in data:
            result.append({
                'input' : d['jobname'],
                'groundtruth' : d['description'] + " " + d['explanation'] if 'description' in d else d['explanation'],
                'source': source,
                'language' : 'ja',
                'task' : 'description',
                'else' : {k:v for k, v in d.items() if k in ['explanation', 'requirement']}
            })
            result.append({
                'input' : d['jobname'],
                'groundtruth' : d['requirements'],
                'source': source,
                'language' : 'ja',
                'task' : 'requirement',
                'else' : {k:v for k, v in d.items() if k in ['explanation', 'requirement']}
            })
    if i == 4:
        for d in data:
            result.append({
                'input' : d['job'],
                'groundtruth' : d['description'],
                'source': source,
                'language' : 'es',
                'task' : 'description',
                'else' : {k:v for k, v in d.items() if k in ['job', 'description']}
            })
    if i == 5:
        for d in data:
            result.append({
                'input' : d['job'],
                'groundtruth' : d['definition']+ " " + d['explanation'],
                'source': source,
                'language' : 'en',
                'task' : 'description',
                'else' : {k:v for k, v in d.items() if k in ['job', 'definition', 'description']}
            })
        
    if i == 6:
        counts = open_json('/Users/sb/Downloads/data5_jobdict_infinigram_count.json')
        en_ko_pairs = [(id, item['en']['count'], item['ko']['jobname'], item['en']['jobname']) for id, item in enumerate(counts)]
        sorted_pairs = sorted(en_ko_pairs, key=lambda x: x[1], reverse=True)
        for (id, cnt, ko, en) in sorted_pairs[:1000]:
            d = data[id]
            result.append({
                'input' : d['jobname']['ko'],
                'groundtruth' : d['definition']+ " " + d['explanation'],
                'source': source,
                'language' : 'ko',
                'task' : 'description',
                'else' : {'count': cnt, **{k:v for k, v in d.items() if k in ['job', 'definition', 'description']}}
            })

            

os.makedirs('data/evalset/', exist_ok=True)
save_json(result, 'data/evalset/truthfulness.jsonl')
print(pd.DataFrame(result)['language'].value_counts())
print(pd.DataFrame(result)['task'].value_counts())