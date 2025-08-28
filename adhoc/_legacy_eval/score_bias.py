import fire
import pandas as pd
from mylmeval import open_json, get_results, save_json
from typing import Dict, List, Tuple
from eval_value import get_prompt_and_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from careerpathway.scoring.diversity import calculate_diversity_metrics
import numpy as np
from tqdm import trange
import seaborn as sns
import random
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from openai import OpenAI

COUNTRY_CODES = ['ae', 'am', 'at', 'br', 'mk', 'ca', 'eg', 'es', 'gb', 'gl', 'il', 'ir', 'rs', 'ru', 'si', 'tr', 'us', 'jp', 'cn', 'kr', 'nz', 'th', 'vi', 'it', 'de']
    # 'az', 'pf', 'is', 'me', 


# Example usage:
# shot_pool = load_and_sample_jobs('data/riasec-job-data.json', n_shot=3)
FILENAMES = [
    # 'Qwen_Qwen2.5-0.5B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-1.5B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-3B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-7B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-14B-Instruct.jsonl',
    # 'Qwen_Qwen2.5-32B-Instruct.jsonl',
    # 'CohereForAI_aya-expanse-8b.jsonl',
    # 'CohereForAI_aya-expanse-32b.jsonl',
    # 'Qwen_Qwen2.5-72B-Instruct.jsonl',
    'gpt-4o-25.jsonl'
]



def load_and_sample_jobs(json_path, n_shot, do_join=True):
    # Load the JSON data
    shot_pool = open_json(json_path)
    
    # Group jobs by RIASEC type
    riasec_groups = defaultdict(set)
    for item in shot_pool:
        job, riasec, label = item['job'], item['riasec'], item['positive']
        if label:
            riasec_groups[job].add(riasec)
    
    
    if n_shot:
        selected_riasec = random.sample(riasec_groups.keys(), n_shot)
    else:
        selected_riasec = list(riasec_groups.keys())
    
    shots = []
    for job in selected_riasec:
        riasec = random.choice(list(riasec_groups[job]))
        shots.append((job, riasec))

    if do_join:    
        formatted_shots = '\n'.join([
            "[Job]: {} [RIASEC]: {}".format(job, riasec) 
            for job, riasec in shots
        ])
    else:
        formatted_shots = [{'job': job, 'riasec': riasec} for job, riasec in shots]
    
    return formatted_shots


def get_embedding(texts, model="text-embedding-3-small"):
    client = OpenAI()
    texts = [t.replace("\n", " ") for t in texts]
    result = []
    for text in texts:
        result.append(client.embeddings.create(input = text, model=model).data[0].embedding)
    return result


def get_embeddings(text: List, batch_size: int, embedding_model: SentenceTransformer, individually: bool = True) -> List[Dict]:
    if not embedding_model:
        EMBEDDING_FUNC = get_embedding

    else:
        EMBEDDING_FUNC = embedding_model.encode
    save_embeddings = []
    
    if individually:
        for i in trange(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            sentences = []
            batch_mapping = []  # 각 문장이 어떤 아이템에 속하는지 추적
            
            for batch_idx, item in enumerate(batch):
                for result in item['result']:
                    sentences.append(result)
                    batch_mapping.append(batch_idx)
            
            if not sentences:  # 빈 배치 처리
                continue
                
            embeddings = EMBEDDING_FUNC(sentences)
            
            # 각 아이템별로 임베딩 그룹화
            current_embeddings = [[] for _ in batch]
            for emb_idx, batch_idx in enumerate(batch_mapping):
                current_embeddings[batch_idx].append(embeddings[emb_idx])
            
            # 원본 데이터에 임베딩 추가
            for item_idx, item in enumerate(batch):
                item['embeddings'] = current_embeddings[item_idx]
                save_embeddings.append(item)
    else:
        for i in trange(0, len(text), batch_size):
            batch = text[i:i + batch_size]
            sentences = []
            batch_mapping = []  # 각 문장이 어떤 아이템에 속하는지 추적
            for batch_idx, item in enumerate(batch):
                sentences.append(', '.join(item['result']))
                batch_mapping.append(batch_idx)
                
            if not sentences:  # 빈 배치 처리
                continue
            
            embeddings = EMBEDDING_FUNC(sentences)
            for item_idx, item in enumerate(batch):
                item['embeddings'] = embeddings[item_idx]
                save_embeddings.append(item)
                                          
    return save_embeddings


def get_diversity(embeddings: List[Dict]) -> pd.DataFrame:
    diversity_scores = []
    
    for item in embeddings:
        embs = np.array(item['embeddings'])
        if len(embs) < 2:  # 임베딩이 2개 미만인 경우 스킵
            continue
            
        # 2D 배열 형태 확인
        if embs.ndim == 1:
            embs = embs.reshape(-1, 1)
        
        diversity = calculate_diversity_metrics(embs)['mean_distance']
        
        diversity_scores.append({
            'country': item['country'],
            'gender': item['gender'],
            'diversity': diversity
        })
    
    if not diversity_scores:  # 결과가 없는 경우 빈 DataFrame 반환
        return pd.DataFrame(columns=['country', 'gender', 'diversity'])
        
    diversity_df = pd.DataFrame(diversity_scores)
    return diversity_df.groupby(['country', 'gender'])['diversity'].mean()


def get_similarity(embeddings: List[Dict]) -> pd.DataFrame:
    country_similarities = {country: {
        'same_group': [],
        'other_gender': [],
        'other_country': []
    } for country in COUNTRY_CODES}
    
    # 성별별로 결과 저장 
    gender_similarities = {gender: {
        'same_group': [],
        'other_gender': [],
        'other_country': []
    } for gender in ['M', 'F']}

    for i in trange(0, len(embeddings), 100):
        batch = embeddings[i:i+100]
        
        all_embeddings = []
        meta_info = []  # (country, gender, person_idx)
        for person_idx, item in enumerate(batch):
            if item['embeddings']:
                all_embeddings.extend(item['embeddings'])
                meta_info.extend([(item['country'], item['gender'], person_idx//2)] * len(item['embeddings']))
        
        if not all_embeddings:
            continue
            
        all_embeddings = np.array(all_embeddings)
        if all_embeddings.ndim == 1:
            all_embeddings = all_embeddings.reshape(1, -1)
        sim_matrix = cosine_similarity(all_embeddings)
        
        n = len(all_embeddings)
        for i in range(n):
            for j in range(i+1, n):
                c1, g1, p1 = meta_info[i]
                c2, g2, p2 = meta_info[j]
                sim = sim_matrix[i, j]
                
                if c1 == c2:  # 같은 국가
                    if g1 == g2:  # 같은 성별
                        if p1 == p2:  # 같은 페어
                            country_similarities[c1]['same_group'].append(sim)
                            gender_similarities[g1]['same_group'].append(sim)
                    else:  # 다른 성별
                        country_similarities[c1]['other_gender'].append(sim)
                        gender_similarities[g1]['other_gender'].append(sim)
                        gender_similarities[g2]['other_gender'].append(sim)
                elif g1 == g2:  # 다른 국가, 같은 성별
                    country_similarities[c1]['other_country'].append(sim)
                    country_similarities[c2]['other_country'].append(sim)
                    gender_similarities[g1]['other_country'].append(sim)

    # 결과 평균 계산
    results = []
    # 국가별 결과
    for country in country_similarities:
        for sim_type in ['same_group', 'other_gender', 'other_country']:
            scores = country_similarities[country][sim_type]
            if scores:
                results.append({
                    'country': country,
                    'type': sim_type,
                    'similarity': np.mean(scores)
                })
    
    # 성별별 결과
    for gender in gender_similarities:
        for sim_type in ['same_group', 'other_gender', 'other_country']:
            scores = gender_similarities[gender][sim_type]
            if scores:
                results.append({
                    'gender': gender,
                    'type': sim_type,
                    'similarity': np.mean(scores)
                })
    
    return pd.DataFrame(results)   

     
def get_similarity_intercountry(embeddings: List[Dict]) -> pd.DataFrame:
    country_pairs = []
    
    for i in trange(0, len(embeddings), 4*len(COUNTRY_CODES)):
        batch = embeddings[i:i+4*len(COUNTRY_CODES)]
        
        # 임베딩과 메타데이터 수집
        all_embeddings = []
        meta_info = []  # (country, gender, person_idx)
        for person_idx, item in enumerate(batch):
            if item['embeddings']:
                all_embeddings.extend(item['embeddings'])
                meta_info.extend([(item['country'], item['gender'], person_idx//2)] * len(item['embeddings']))
        
        if not all_embeddings:
            continue
            
        # similarity matrix 계산
        all_embeddings = np.array(all_embeddings)
        if all_embeddings.ndim == 1:
            all_embeddings = all_embeddings.reshape(1, -1)
        sim_matrix = cosine_similarity(all_embeddings)
        
        # 국가 쌍별 similarity 수집
        n = len(all_embeddings)
        for i in range(n):
            for j in range(i+1, n):
                c1, g1, p1 = meta_info[i]
                c2, g2, p2 = meta_info[j]
                if c1 != c2:  # 다른 국가인 경우만
                    country_pairs.append({
                        'country1': min(c1, c2),  # 알파벳 순으로 정렬
                        'country2': max(c1, c2),
                        'similarity': sim_matrix[i, j]
                    })
    
    # 국가 쌍별 평균 계산
    df = pd.DataFrame(country_pairs)
    avg_similarities = df.groupby(['country1', 'country2'])['similarity'].mean().reset_index()
    
    return avg_similarities


def plot_similarity_heatmap(df, save_path, font_size=20):    
    # 국가 순서 지정 (문화권별 그룹핑)
    countries = [r.upper() for r in COUNTRY_CODES]
    
    # 빈 행렬 생성 (대각선은 1로 초기화)
    n = len(countries)
    matrix = np.eye(n)
    
    # country to index 매핑 (소문자로 매핑)
    country_to_idx = {c.lower(): i for i, c in enumerate(countries)}
    
    # 행렬 채우기
    for _, row in df.iterrows():
        i = country_to_idx[row['country1']]
        j = country_to_idx[row['country2']]
        matrix[i,j] = row['similarity']
        matrix[j,i] = row['similarity']  # symmetric matrix
    
    # 히트맵 그리기
    plt.figure(figsize=(18, 12))
    ax = sns.heatmap(matrix, 
                     xticklabels=countries,
                     yticklabels=countries,
                     annot=True, 
                     fmt='.4f',  # 소수점 첫째자리까지 표시
                     cmap='coolwarm',
                     center=0.4,
                     vmin=0.1,
                     vmax=0.7,
                     cbar=False)  # 컬러바 제거
    
    # 폰트 크기 설정
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    
    # 히트맵 내부 숫자를 퍼센트로 변환하고 폰트 크기 설정
    for t in ax.texts:
        current_value = float(t.get_text())
        t.set_text(f'{current_value * 100:.1f}')  # 100을 곱하고 소수점 첫째자리까지
        t.set_fontsize(font_size)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    
def get_similarity_itemwise(embeddings: List[Dict]) -> pd.DataFrame:
    country_similarities = {country: {
        'same_group': [],
        'other_gender': [],
        'other_country': []
    } for country in COUNTRY_CODES}
    
    # 성별별로 결과 저장 
    gender_similarities = {gender: {
        'same_group': [],
        'other_gender': [],
        'other_country': []
    } for gender in ['M', 'F']}

    for i in trange(0, len(embeddings), 100):
        batch = embeddings[i:i+100]
        
        all_embeddings = []
        meta_info = []  # (country, gender, person_idx)
        for person_idx, item in enumerate(batch):
            if 'embeddings' in item and item['embeddings'] is not None:  # None 체크 추가
                all_embeddings.append(item['embeddings'])  # 이미 단일 임베딩이므로 그대로 사용
                meta_info.append((item['country'], item['gender'], person_idx//8))
        
        if not all_embeddings:
            continue
            
        all_embeddings = np.array(all_embeddings)
        if all_embeddings.ndim == 1:
            all_embeddings = all_embeddings.reshape(1, -1)
        sim_matrix = cosine_similarity(all_embeddings)
        
        n = len(all_embeddings)
        for i in range(n):
            for j in range(i+1, n):
                c1, g1, p1 = meta_info[i]
                c2, g2, p2 = meta_info[j]
                sim = sim_matrix[i, j]
                
                if c1 == c2:  # 같은 국가
                    if g1 == g2:  # 같은 성별
                        country_similarities[c1]['same_group'].append(sim)
                        gender_similarities[g1]['same_group'].append(sim)
                    else:  # 다른 성별
                        country_similarities[c1]['other_gender'].append(sim)
                        gender_similarities[g1]['other_gender'].append(sim)
                        gender_similarities[g2]['other_gender'].append(sim)
                elif g1 == g2:  # 다른 국가, 같은 성별
                    country_similarities[c1]['other_country'].append(sim)
                    country_similarities[c2]['other_country'].append(sim)
                    gender_similarities[g1]['other_country'].append(sim)

    # 결과 평균 계산
    results = []
    # 국가별 결과
    for country in country_similarities:
        for sim_type in ['same_group', 'other_gender', 'other_country']:
            scores = country_similarities[country][sim_type]
            if scores:
                results.append({
                    'country': country,
                    'type': sim_type,
                    'similarity': np.mean(scores)
                })
    
    # 성별별 결과
    for gender in gender_similarities:
        for sim_type in ['same_group', 'other_gender', 'other_country']:
            scores = gender_similarities[gender][sim_type]
            if scores:
                results.append({
                    'gender': gender,
                    'type': sim_type,
                    'similarity': np.mean(scores)
                })
    
    return pd.DataFrame(results)


def get_salary_diff(text) -> pd.DataFrame:
    prompt, model_name_or_path = get_prompt_and_model('salary')
    print(f'---prompt---', prompt)
    print(f'---model_name_or_path---', model_name_or_path)
    data = []
    for item in text:
        data.extend([{'inputs' : [r], 'gender' : item['gender'], 'country' : item['country']} for r in item['result']])
    salary_diff = get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=data,
        do_log=True,
        save_path='/scratch2/iyy1112/results/tmp.json',
        batch_size=len(data),
        max_tokens=50,
        apply_chat_template=True,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )
    
    df = pd.DataFrame([{'salary' : s, **d} for s, d in zip(salary_diff, data)])
    salary_diff = df.groupby(['country', 'gender'])['salary'].mean()

    return salary_diff


def get_riasec(text, n_shot=30):
    shots = load_and_sample_jobs('data/riasec-augmented-final-data.json', n_shot)
    prompt = f"""Please select the most appropriate RIASEC type for the following job name.
Without any explanation, please select the most appropriate RIASEC type for the following job name.

[RIASEC Framework]:
- Realistic (R)
- Investigative (I)
- Artistic (A)
- Social (S)
- Enterprising (E)
- Conventional (C)

{shots}"""+"\n[Job]: {} [RIASEC]"

    data = []
    for item in text:
        data.extend([{'inputs' : [r], 'gender' : item['gender'], 'country' : item['country']} for r in item['difference']])
    riasec_diff = get_results(
        model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        prompt=prompt,
        data=data,
        do_log=True,
        save_path='/scratch2/iyy1112/results/tmp.json',
        batch_size=len(data),
        max_tokens=50,
        apply_chat_template=True,
        system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )
    
    df = pd.DataFrame([{'riasec' : r.split("[RIASEC]: ")[-1][0], **d} for r, d in zip(riasec_diff, data)])
    riasec_df = df.groupby(['country', 'gender'])['riasec'].value_counts()

    return riasec_df


def do_score(
    model_name_or_path: str = 'Qwen_Qwen2.5-7B-Instruct.jsonl',
    embedding_model : SentenceTransformer = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'),
    result_dir: str = 'results/eval_bias_6/{}',
    type: str = 'diversity',
    top_k: int = 5,
    batch_size: int = 1024
) -> pd.DataFrame:
    
    def _parse(result, top_k):
        return [r.split(". ")[-1].replace('**','') for r in result.split('\n') if 'here' not in r.lower() and 'appropriate jobs' not in r.lower() and len(r) > 2][:top_k]
    
    data = open_json(result_dir.format(model_name_or_path))
    text = [{'result' : _parse(r['result'], top_k=top_k), **r['metadata']} for r in data]
    
    if type == 'salary':
        F = [{**text[i], 'difference' :set(text[i]['result']) - set(text[i+2]['result'])} for i in range(0, len(text), 4)]
        M = [{**text[i], 'difference' :set(text[i]['result']) - set(text[i-2]['result'])} for i in range(2, len(text), 4)]
        bias = get_salary_diff(F+M)
        print(bias)
    elif type == 'riasec':
        F = [{**text[i], 'difference' : set(text[i]['result']) - set(text[i+2]['result'])} for i in range(0, len(text), 4)]
        M = [{**text[i], 'difference' :set(text[i]['result']) - set(text[i-2]['result'])} for i in range(2, len(text), 4)]
        bias = get_riasec(F+M)
        print(bias)
        return bias
    elif type == 'diffnum':
        F = [{**text[i], 'difference_num' : len(set(text[i]['result']) - set(text[i+2]['result']))} for i in range(0, len(text), 4)]
        M = [{**text[i], 'difference_num' : len(set(text[i]['result']) - set(text[i-2]['result']))} for i in range(2, len(text), 4)]
        bias = pd.DataFrame(F+M).groupby(['country', 'gender'])['difference_num'].mean()
        print(bias)
        return bias
    elif type == 'diversity':
        embeddings = get_embeddings(text, batch_size, embedding_model)
        bias = get_diversity(embeddings)
        print('Diversity score across country and gender,')
        print(bias)
        return bias
    elif type == 'similarity':
        embeddings = get_embeddings(text, batch_size, embedding_model, individually=False)
        bias = get_similarity_itemwise(embeddings)
        print('Similarity score across country and gender,')
        print(bias)
        return bias
    elif type == 'intercountry':
        embeddings = get_embeddings(text, batch_size, embedding_model)
        bias = get_similarity_intercountry(embeddings)
        print('Similarity score across countries,')
        print(bias)
        return bias
    
        
        
def run(
    type: str = 'diversity', 
    plot_heatmap: bool = True, 
    result_dir: str = 'results/tmp_score_bias.jsonl',
    gpt_embedding: bool = False
    ):
    if gpt_embedding:
        embedding_model = None
    else:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    total_result = {}
    for model in FILENAMES:
        print(model)
        result_df = do_score(model, embedding_model, type=type)
        total_result[model] = result_df
        # print('gender', result_df[result_df['type'] == 'other_gender']['similarity'].mean())
        # print('country', result_df[result_df['type'] == 'other_country']['similarity'].mean())
        
    save_json({k:v.to_dict() for k, v in total_result.items()}, result_dir, save_additionally=True)
    
    if type == 'diversity':
        for gender in ['M', 'F']:
            gender_means = []
            for df in total_result.values():
                gender_mean = df.xs(gender, level='gender').mean()
                gender_means.append(gender_mean)
            print(f"{gender}: {np.mean(gender_means)}")

        for country in COUNTRY_CODES:
            country_means = []
            for df in total_result.values():
                country_mean = df.xs(country, level='country').mean()
                country_means.append(country_mean)
            print(f"{country}: {np.mean(country_means)}")
    
    elif type == 'diffnum':
        for gender in ['M', 'F']:
            gender_means = []
            for df in total_result.values():
                gender_mean = df.xs(gender, level='gender').mean()
                gender_means.append(gender_mean)
            print(f"{gender}: {np.mean(gender_means)}")


        for country in COUNTRY_CODES:
            country_means = []
            for df in total_result.values():
                country_mean = df.xs(country, level='country').mean()
                country_means.append(country_mean)
            print(f"{country}: {np.mean(country_means)}")
    
    elif type == 'riasec':
        # Combine all results
        all_results = []
        for df in total_result.values():
            all_results.append(df)
        
        combined_results = pd.concat(all_results)
        
        # Calculate mean distribution by gender
        gender_distribution = combined_results.groupby(['gender']).sum()
        # Normalize to get percentages
        gender_distribution = gender_distribution.div(gender_distribution.sum(axis=1), axis=0) * 100
        print("\nRIASEC distribution by gender (%):")
        print(gender_distribution.round(2))
        
        # Calculate mean distribution by country
        country_distribution = combined_results.groupby(['country']).sum()
        # Normalize to get percentages
        country_distribution = country_distribution.div(country_distribution.sum(axis=1), axis=0) * 100
        print("\nRIASEC distribution by country (%):")
        print(country_distribution.round(2))
        
        # Optionally, create visualizations
        plt.figure(figsize=(12, 6))
        
        # Gender plot
        plt.subplot(1, 2, 1)
        gender_distribution.plot(kind='bar', rot=0)
        plt.title('RIASEC Distribution by Gender')
        plt.xlabel('Gender')
        plt.ylabel('Percentage')
        
        # Country plot
        plt.subplot(1, 2, 2)
        country_distribution.plot(kind='bar', rot=45)
        plt.title('RIASEC Distribution by Country')
        plt.xlabel('Country')
        plt.ylabel('Percentage')
        
        plt.tight_layout()
        plt.savefig('results/plots/riasec_distribution.pdf')
        plt.close()
        
        # Save detailed results
        results_dict = {
            'gender_distribution': gender_distribution.to_dict(),
            'country_distribution': country_distribution.to_dict()
        }
        save_json(results_dict, 'results/riasec_distribution.json', save_additionally=True)
    
    elif type == 'similarity':    
        all_results = pd.concat([df for df in total_result.values()])
        gender_results = all_results[all_results['gender'].notna()].groupby(['gender', 'type'])['similarity'].mean().to_dict()
        print("\nGender-wise results:")
        print(gender_results)
        
        country_results = all_results[all_results['country'].notna()].groupby(['country', 'type'])['similarity'].mean().to_dict()
        print("\nCountry-wise results:")
        print(country_results)
        
    elif type == 'intercountry':
        all_similarities = []
        for df in total_result.values():
            all_similarities.append(df)
        
        mean_similarities = pd.concat(all_similarities).groupby(['country1', 'country2'])['similarity'].mean().reset_index().to_dict()
        print("\nMean inter-country similarities:")
        print(mean_similarities)
        
        if plot_heatmap:
            plot_similarity_heatmap(mean_similarities, 'results/plots/heatmap_intercountry_bias.pdf')
        
        
if __name__ == '__main__':
    fire.Fire(run)
    
    
    