from mylmeval import get_results, open_json, save_json
import fire
import pandas as pd
import os
import random
from collections import Counter

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"]="1"

MODEL_NAMES = [
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Meta-Llama-3-70B-Instruct',
]

PROMPTS = {
    'en' : """Please answer the best appropriate job title for the following job requirements.
[Job Requirements]: {}
[Appropriate Job]:""",
    'es' : """Por favor, responda el mejor título de trabajo apropiado para los siguientes requisitos de trabajo.
[Requisitos de trabajo]: {}
[Trabajo apropiado]:""",
    'ja' : """次の求人要件に最適な適切な職名を回答してください。
[求人要件]: {}
[適切な職名]:""",
    'ko' : """다음 직무 요구 사항에 가장 적합한 직업을 선택하십시오.
[직무 요구 사항]: {}
[적합한 직업]:"""
    }


INPUT_FORMATS = {
    'en' : """Job Description: {}\nJob Skills: {}""",
    'es' : """Descripción del trabajo: {}\nHabilidades laborales: {}""",
    'ja' : """仕事の説明: {}\n仕事のスキル: {}""",
    'ko' : """직무 설명: {}\n직무 기술: {}"""
    }


def _jaccard_similarity(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def _get_most_similar_jobs(data, es_data_jobs, n=1000):
    # es_data_jobs를 하나의 문자열로 합칩니다
    es_data_jobs_str = " ".join(es_data_jobs)
    
    # 각 작업에 대해 Jaccard 유사도를 계산합니다
    similarities = []
    for job in data:
        similarity = _jaccard_similarity(job['jobname']['es'].lower(), es_data_jobs_str)
        similarities.append((job, similarity))
    
    # 유사도에 따라 정렬하고 상위 n개를 선택합니다
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 중복을 제거하면서 상위 n개의 작업을 선택합니다
    selected_jobs = []
    seen_jobnames = set()
    for job, _ in similarities:
        jobname = job['jobname']['es'].lower()
        if jobname not in seen_jobnames:
            selected_jobs.append(job)
            seen_jobnames.add(jobname)
        if len(selected_jobs) == n:
            break
    
    return selected_jobs


def process_data(source_lang='en', translated_lang='en', input_format='{}'):
    random.seed(42)
    if (source_lang, translated_lang) == ('en', 'en'):
        data = open_json('data/data13_15_kaggle_sample.json')
        data = [r for r in data if r['jobskills'] != None]
        data = random.sample(data, 937)
        data = [{
            'inputs':[input_format.format(r['description'], r['jobskills'])], 
            'groundtruth': r['jobname'], 
            'source_lang' : 'en'
                } for r in data]
    elif (source_lang, translated_lang) == ('es', 'es'):
        data = open_json('data/data7_educaweb.json')
        data = [{
            'inputs' : [input_format.format(r['description'], r['requirements'] if 'requirements' in r else 'unknown')], 
            'groundtruth': r['job'],
            'source_lang' : 'es'
            } for r in data]
    elif (source_lang, translated_lang) == ('ja', 'ja'):
        data = open_json('data/data11_onet.json')
        data = [{
            'inputs':[input_format.format(r['explanation'], r['requirements'])],
            'groundtruth': r['jobname'],
            'source_lang' : 'ja'
            } for r in data]
    elif (source_lang, translated_lang) == ('ko', 'ko'):
        es_data_jobs = [r['job'].lower() for r in open_json('data/data7_educaweb.json')]
        data = open_json('data/data5_jobdict.json')
        similar_jobs = _get_most_similar_jobs(data, es_data_jobs, 1000)
        data = pd.DataFrame(similar_jobs).drop_duplicates('jobname').sample(len(es_data_jobs), random_state=42).to_dict('records')
        data = [{
            'inputs' : [input_format.format(r["explanation"], r['info']['자격/면허'])], 
            'groundtruth': r['jobname'],
            'lang': 'ko'
            } for r in data]
    return data


def load_data(data_path: str = 'data/infer_4lang.jsonl'):
    try:
        data = open_json(data_path)
    except:
        result = []
        for lang in ['en', 'es', 'ja', 'ko']:
            prompt = PROMPTS[lang]
            data = _process_data(lang, lang, INPUT_FORMATS[lang])
            result.extend([{'inputs':[prompt.format(*r['inputs'])], 'groundtruth':r['groundtruth']} for r in data])
        print(len(result))
        save_json(result, data_path)
    return data


def infer_llm(
    model_name_or_path: int = MODEL_NAMES[1],
    source_lang: str = 'en',
    translated_lang: str = 'en',
    ):
    data = load_data()
    _ = get_results(
            model_name_or_path=model_name_or_path,
            prompt='{}',
            data=data,
            do_log=True,
            batch_size=len(data),
            max_tokens=64,
            apply_chat_template=True,
            system_prompt=None,
            save_path=f"results/{model_name_or_path.replace('/', '_')}_{source_lang}_{translated_lang}_infer.jsonl"
            )
    print(f"Results saved in results/{model_name_or_path}_kaggle_infer.json")



if __name__ == '__main__':
    fire.Fire(infer_llm)
