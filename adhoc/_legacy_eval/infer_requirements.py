from mylmeval import get_results, open_json, save_json
import fire
import pandas as pd
import os
import random
from collections import Counter
import matplotlib.pyplot as plt

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"]="1"

MODEL_NAMES = [
    'Qwen/Qwen2.5-72B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'meta-llama/Llama-3-70B-Instruct',
]

PROMPTS = {
    'ko' : "다음 직업과 관련 있는 1. 직무 적합성과 2. 요구사항을 설명하세요. 설명을 길게 하지 말고, 중요한 내용만 간략하게 답하세요.\n직업: {}\n직무 적합성:",
    'es' : "Describa la idoneidad laboral y los requisitos relacionados con el siguiente trabajo. No se extienda, solo responda brevemente los puntos clave.\nTrabajo: {}\nIdoneidad laboral:",
    'en' : "Describe the job fitness and requirements related to the following job. Do not elaborate, just briefly answer the key points.\nJob: {}\nJob Fitness:",
    'ja' : "次の仕事に関連する仕事適性と要件を説明してください。詳しくは説明せず、重要なポイントに簡潔に回答してください。\n仕事: {}\n仕事適性:"
}

INPUT_FORMATS = {
    'en' : """Job Description: {}\nJob Skills: {}""",
    'es' : """Descripción del trabajo: {}\nHabilidades laborales: {}""",
    'ja' : """仕事の説明: {}\n仕事のスキル: {}""",
    'ko' : """직무 설명: {}\n직무 기술: {}"""
    }


import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# NLTK 데이터 다운로드
nltk.download('punkt_tab')

def calculate_bleu(reference, hypothesis):
    """
    BLEU 스코어 계산
    
    :param reference: str, 참조 텍스트
    :param hypothesis: str, 생성된 텍스트
    :return: float, BLEU 스코어
    """
    reference_tokens = word_tokenize(reference.lower())
    hypothesis_tokens = word_tokenize(hypothesis.lower())
    
    return sentence_bleu([reference_tokens], hypothesis_tokens)

def calculate_meteor(reference, hypothesis):
    """
    METEOR 스코어 계산
    
    :param reference: str, 참조 텍스트
    :param hypothesis: str, 생성된 텍스트
    :return: float, METEOR 스코어
    """
    reference_tokens = word_tokenize(reference.lower())
    hypothesis_tokens = word_tokenize(hypothesis.lower())
    
    return meteor_score([reference_tokens], hypothesis_tokens)



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


def _process_data(lang='en', input_format='{}'):
    random.seed(42)
    if lang == 'en':
        pass
    elif lang == 'es':
        data = open_json('data/data7_educaweb.json')
        data = [r for r in data if 'job' in r and 'requirements' in r and 'fitness' in r]
        data = random.sample(data, 500)
        data = [{
            'inputs' : [input_format.format(r['job'])], 
            'groundtruth': [r['fitness'], r['requirements']],
            'lang' : 'es'
            } for r in data]
    elif lang == 'ja':
        pass
    elif lang == 'ko':
        data = open_json('data/data16_ko_essensial.json')
        data = random.sample(data, 500)
        data = [{
            'inputs':[input_format.format(r['직업명'])],
            'groundtruth': [
                ', '.join([r.split("(")[0] for r in (r['지식']+r['환경']+r['성격']).split("\n")]),
                r['필요기술 및 지식']
                ],
        } for r in data]
    return data


def load_data(data_path: str = 'data/job_to_requirements.jsonl'):
    try:
        data = open_json(data_path)
    except:
        data = []
        for lang in ['es', 'ko']:
            prompt = PROMPTS[lang]
            lang_data = _process_data(lang, PROMPTS[lang])
            data.extend(lang_data)
        print(len(data))
        save_json(data, data_path)
    return data


def infer_llm(
    model_name_or_path: int = 'gpt-4o-mini',
    gather_score: bool = False,
    start: int = 0,
    ):
    if gather_score:
        data = open_json(f'results/{model_name_or_path.replace("/", ",")}_infer.jsonl')
        scores = {'bleu':[], 'meteor':[]}
        for i, lang in enumerate(['es', 'ko']):
            for d in data[i*500:(i+1)*500]:
                blue_score = calculate_bleu('\n'.join(d['groundtruth']), d['result'])
                meteor_score = calculate_meteor('\n'.join(d['groundtruth']), d['result'])
                scores['bleu'].append(blue_score)
                scores['meteor'].append(meteor_score)
            print(f"BLEU: {sum(scores['bleu'])/len(scores['bleu'])}")
            print(f"METEOR: {sum(scores['meteor'])/len(scores['meteor'])}")
            # plot
            fig, ax = plt.subplots()  # figure와 axes 동시에 생성
            ax.hist(scores['bleu'], bins=20, alpha=0.5, label='BLEU')
            ax.hist(scores['meteor'], bins=20, alpha=0.5, label='METEOR')
            ax.legend(loc='upper right')
            ax.set_title(f"{model_name_or_path} {lang} BLEU: {sum(scores['bleu'])/len(scores['bleu']):.4f} METEOR: {sum(scores['meteor'])/len(scores['meteor']):.4f}")
            fig.savefig(f"results/{model_name_or_path.replace('/', '_')}_{lang}_score.png")
        return
        
    data = load_data()[start:]
    _ = get_results(
            model_name_or_path=model_name_or_path,
            prompt='{}',
            data=data,
            do_log=True,
            batch_size=len(data),
            max_tokens=256,
            apply_chat_template='auto',
            system_prompt=None,
            save_path=f"results/{model_name_or_path.replace('/', ',')}_infer_{start}.jsonl"
            )
    print("Done!")






if __name__ == '__main__':
    fire.Fire(infer_llm)
