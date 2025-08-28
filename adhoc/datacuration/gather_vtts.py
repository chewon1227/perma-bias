import json
from tqdm import tqdm
import io
import os
import fire
import datetime
import pandas as pd
import re
from mylmeval import save_json
import unicodedata

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

def clean_text(text):
    text = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', text)
    text = re.sub(r'</?c>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def list_into_oneline(lst):
    result = ''
    for i in range(len(lst))[::-1]:
        if lst[i] not in result:
            result = lst[i] + ' ' + result
    return result
        
        
def parse_vtt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    reader = io.StringIO(content)
    subtitles = []
    next(reader)

    current_text = ""
    for line in reader:
        line = line.strip()
        
        if '-->' in line:
            if current_text:
                subtitles.append(clean_text(current_text))
                current_text = ""
            start_time = line.split(' --> ')[0]
        elif line:
            current_text += " " + line

    # Add the last subtitle if exists
    if current_text:
        subtitles.append(clean_text(current_text))

    return list_into_oneline(subtitles)


def combine_jamo(text):
    result = []
    current_syllable = []
    
    # 한글 자모 유니코드 범위
    CHOSEONG_START, CHOSEONG_END = 0x1100, 0x1112
    JUNGSEONG_START, JUNGSEONG_END = 0x1161, 0x1175
    JONGSEONG_START, JONGSEONG_END = 0x11A8, 0x11C2

    for char in text:
        if (CHOSEONG_START <= ord(char) <= CHOSEONG_END or
            JUNGSEONG_START <= ord(char) <= JUNGSEONG_END or
            JONGSEONG_START <= ord(char) <= JONGSEONG_END):
            current_syllable.append(char)
        else:
            if current_syllable:
                result.append(unicodedata.normalize('NFC', ''.join(current_syllable)))
                current_syllable = []
            result.append(char)
    
    if current_syllable:
        result.append(unicodedata.normalize('NFC', ''.join(current_syllable)))
    
    return ''.join(result)


def normalize_korean(text):
    return combine_jamo(unicodedata.normalize('NFD', text))


def preprocess_text(text):
    # 한글 정규화 및 자모 결합
    text = normalize_korean(text)
    # 모든 특수 문자를 제거하고 소문자로 변환
    text = re.sub(r'[^\w\s가-힣]', '', text.lower())
    return text


def create_jobid_dict(jobid_data):
    # 전처리된 title을 키로, (id, job)을 값으로 하는 딕셔너리 생성
    return {preprocess_text(row['title']): (row['idx'], row['job']) for _, row in jobid_data.iterrows()}


def match_jobid(jobid_dict, title, lang, real_jobdict):
    if '' in jobid_dict:
        del jobid_dict['']
        
    processed_title = preprocess_text(title)[:-1] # for korean
    
    # 완전 일치 검사
    if processed_title in jobid_dict:
        jobid = jobid_dict[processed_title][0]
        return jobid, real_jobdict[jobid]['jobname'][lang]
    
    # 부분 일치 검사
    for key, value in jobid_dict.items():
        if processed_title in key or key in processed_title:
            return value[0], real_jobdict[value[0]]['jobname'][lang]
    
    return None, None


def gather_vtts(
    vtts_path: str = '/scratch2/sb/vtts',
    ):
    result = []
    video_titles = []
    dedup = 0
    jobid_data = create_jobid_dict(pd.read_csv('data/urls.csv', sep='\t'))
    real_jobdict = pd.read_json('data/data5_jobdict.json').to_dict(orient='records')
    
    print(len(os.listdir(vtts_path)))
    for vtt_file in tqdm(os.listdir(vtts_path)):
        lang = vtt_file[-6:-4]
        if lang not in ['es', 'en', 'ko', 'ja']:
            vtt_file = vtt_file.split(".vtt")[0] + ".vtt"
            lang = vtt_file[-6:-4]
        title = vtt_file[:-21]
        code = vtt_file[-19:-8]
        id, job = match_jobid(jobid_data, title, lang, real_jobdict)
        if id is None:
            print(f"Job ID not found for {title}")
        if title in video_titles:
            dedup += 1
        else:
            try:
                parsed_text = parse_vtt(f'{vtts_path}/{vtt_file}').replace(f'Kind: captions Language: {lang} ', '')
            except:
                print(f'Error in {vtts_path}/{vtt_file}')
                continue
            result.append(
                {'language' : lang, 'title' : title, 'code' : code, 'job' : job, 'text' : parsed_text}
                )
            video_titles.append(title)
        save_json(result, f"vtts_{now}.json", save_every=5000)
    
    print(pd.DataFrame(result)['language'].value_counts())
    print(f"Found {len(result)} vtts, {dedup} duplicates")



def check_progress(
    vtts_path: str = '/scratch2/sb/vtts'
    ):
    done_titles = [r[:-21] for r in os.listdir(vtts_path)]

    shards = {
        'es_0': 'data/es_0_dedup.csv',
        'es_1': 'data/es_1_dedup.csv',
        'en_0': 'data/en_0_dedup.csv',
        'en_1': 'data/en_1_dedup.csv',
        'ko_0': 'data/ko_0_dedup.csv',
        'ko_1': 'data/ko_1_dedup.csv',
        'ja_0': 'data/ja_0_dedup.csv',
        'ja_1': 'data/ja_1_dedup.csv'
    }
    
    for shard_name, shard_path in shards.items():
        undone = []
        # 청크 단위로 CSV 파일 읽기
        for chunk in pd.read_csv(shard_path, chunksize=10000):
            chunk_titles = chunk['title'].tolist()
            unprocessed_mask = ~chunk['title'].isin(done_titles)
            unprocessed_rows = chunk[unprocessed_mask]
            
            undone.extend(unprocessed_rows.to_dict(orient='records'))
            
            processed_titles = set(chunk_titles) & set(done_titles)
            last_done_index = max([chunk_titles.index(title) for title in processed_titles] + [-1])
            
            # if last_done_index < len(chunk_titles) - 1:
                # undone.extend(chunk.iloc[last_done_index + 1:].to_dict(orient='records'))
            
            progress_percentage = (len(processed_titles) / len(chunk_titles)) * 100
            print(f"{shard_name} progress (up to row {chunk.index[-1]}): {progress_percentage:.2f}%, last done index: {last_done_index}")
        
        # save csv
        pd.DataFrame(undone).to_csv(f"{shard_path.replace('.csv', '_undone.csv')}", index=False)

            
            
if __name__ == '__main__':
    fire.Fire(gather_vtts)
    # df = pd.read_json('/home/iyy1112/workspace/Career-Pathway/vtts_2024-10-13_21-02.json')
    # import pdb; pdb.set_trace()
    