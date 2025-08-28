import json
import requests
from tqdm import tqdm
from time import sleep

with open('data/data5_jobdict.json', 'r') as f:
    jobdict = json.load(f)

def get_infinigram_count(jobname):
    payload = {
        'index': 'v4_dolma-v1_7_llama',
        'query_type': 'count',
        'query': jobname,
    }
    result = requests.post('https://api.infini-gram.io/', json=payload).json()
    return result['count']

job_count_list = []

for job in tqdm(jobdict):
    jobname_en = job['jobname']['en']
    jobname_en_lower = jobname_en.lower()
    jobname_en_capitalized = jobname_en[0] + jobname_en_lower[1:]
    jobname_ko = job['jobname']['ko']
    while True:
        try:
            count_en = get_infinigram_count(jobname_en)
            count_en_lower = get_infinigram_count(jobname_en_lower)
            count_en_capitalized = get_infinigram_count(jobname_en_capitalized)
            count_ko = get_infinigram_count(jobname_ko)
            break
        except:
            sleep(1)

    job_count_list.append({
        'idx': job['idx'],
        'en': {'jobname': jobname_en, 'count': count_en},
        'en_lower': {'jobname': jobname_en_lower, 'count': count_en_lower},
        'en_capitalized': {'jobname': jobname_en_capitalized, 'count': count_en_capitalized},
        'ko': {'jobname': jobname_ko, 'count': count_ko},
    })

with open('data/data5_jobdict_infinigram_count.json', 'w') as f:
    json.dump(job_count_list, f, indent=2, ensure_ascii=False)
    