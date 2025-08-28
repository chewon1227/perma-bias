import fire
from mylmeval import open_json, save_json, get_results
from tqdm import tqdm
import random
import os

def remove_immediate_duplicates(text):
    words = text.split()
    result = []
    
    i = 0
    while i < len(words):
        result.append(words[i])
        for length in range(min(20, len(words) - i), 0, -1):  # 최대 20단어 길이의 패턴까지 확인
            pattern = words[i:i+length]
            if i + length < len(words) and words[i+length:i+length*2] == pattern:
                i += length
                break
        i += 1
    print(f"Original: {len(text)}, Processed: {len(' '.join(result))}")
    return ' '.join(result)



def title_classifying_prompt(
    shot=40, 
    data=None
    ):
    neg = random.sample([r for r in data if r['output'] == 'Negative'], shot//2)
    pos = random.sample([r for r in data if r['output'] == 'Positive'], shot//2)
    
    shots = "\n\n".join([f"{r['input']}\n[Label]: {r['output']}" for r in neg+pos])
    return """Please classify if a youtube video with this title will have information of that job.
Without any extra words, please reply with 'Positive' or 'Negative'.

{}

[Video Title]: {}
[Job Name]: {}
[Label]:""".format(shots, '{}', '{}')
    

def extract_info_prompt():
    return """Analyze the YouTube video content (title and transcript) to extract career information, following this structured format:

### Youtube Video Content
Title: {}
Transcript: {}
Questions to answer:

### Career Information
1. Does this content discuss {}? (Yes/No)
2. What other the most related career discussed in the content? 
3. For any career discussed, provide information in this format:
Use "~~" if information is missing.
  definition: [core responsibilities, main function]
  requirements:
    - education
    - skills
    - experience
  compensation:
    - salary range
    - benefits
4. List all background/qualifications mentioned:
Use "~~" if information is missing.
    Education (including specialized programs)
    Previous roles/experiences
    Skills/certifications
    Notable achievements

### Notes
1. Include only explicitly mentioned information.
2. Use exact numbers/dates when stated.
3. Keep chronological order for background information

### Answer"""


def process(
    vtt_file_path: str = 'results/ytb_temp/title_classifier_positive.json',
    tmp_file_path: str = 'results/ytb_temp/tmp.json',
    model_name_or_path: str = 'Qwen/Qwen2.5-32B-Instruct',
    stage: float = 0,
    start: int | None = None,
    shot: int = 80,
    how_many: int = 5000
    ):
    data = open_json(vtt_file_path)
    fewshots = open_json('data/title_classifier.json')
    original_num = len(data)
    
    if stage == 0:
        # 1. length filtering
        data = [d for d in data if len(d['text']) > 200]
        print(f"Filtered: {len(data)/original_num:.2f}, Left: {len(data)}")
        save_json(data, tmp_file_path)
    
    if stage == 1:
        # 2. basic processing
        for d in tqdm(data):
            d['text'] = remove_immediate_duplicates(d['text'])
        save_json(data, tmp_file_path)
    
    if stage == 2:
        result = get_results(
            model_name_or_path=model_name_or_path,
            prompt=title_classifying_prompt(shot, fewshots),
            data=[{'inputs' : [d['title']+d['code'], d['job']]} for d in data][start:start+how_many],
            max_tokens=10,
            batch_size=len(data),
            save_path=tmp_file_path.replace('.json', f'_title_{start}.json'),
            do_log=True,
            )
        new_data = [d for d, r in zip(data, result) if r['result'] == 'Positive']
        print(f"Filtered: {len(new_data)/original_num:.2f}, Left: {len(new_data)}")
        save_json(new_data, tmp_file_path.replace('.json', f'_title_{start}.json'))
        data = new_data
        
    if stage == 2.5:
        files = [r for r in os.listdir('.') if 'title_' in r]
        print(files)
        data = []
        for f in files:
            f_data = open_json(f)
            data += f_data
        print(f"Total: {len(data)}")
        print(f"Positive: {len([d for d in data if 'Positive' in d['result']])}")
        print(f"Negative: {len([d for d in data if 'Negative' in d['result']])}")
        save_json([d for d in data if 'Positive' in d['result']], 'data/title_classifier_positive.json')
        for f in files:
            os.remove(f)
            
    if stage == 3:
        data = [{'inputs' : [d['title']+d['code'], d['text'], d['job']]} for d in data]
        results = get_results(
            model_name_or_path=model_name_or_path,
            prompt=extract_info_prompt(),
            data=data[start:start+how_many] if start else data,
            max_tokens=2048,
            batch_size=len(data),
            save_path=tmp_file_path.replace('.json', f'_extract_{start}.json'),
            do_log=True,
            )
        save_json(results, tmp_file_path.replace('.json', f'{start}_extract.json'))
        
    
    
        
    
if __name__ == '__main__':
    fire.Fire(process)