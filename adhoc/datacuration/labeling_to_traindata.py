import pandas as pd
import json
from tqdm import tqdm
data_dir = [
    '~/Downloads/es_label.json',
    '~/Downloads/en_label.json',
    '~/Downloads/ja_label.json',
    '~/Downloads/ko_label.json'
]

result = []
instruction = 'Please classify if a youtube video with this title will have information of that job.'
input_format = '[Video Title]: {}\n[Job Name]: {}'

for data_path in tqdm(data_dir):
    df = pd.read_json(data_path)
    for i, row in df.iterrows():
        try:
            label = row['annotations'][0]['result'][0]['value']['choices'][0]
            print(label)
        except:
            continue
        if label in ['Positive', 'Negative']:
            result.append(
                {
                    'instruction' : instruction,
                    'input' : input_format.format(row['data']['title'], row['data']['job']),
                    'output' : label
                }
            )
    

df = pd.DataFrame(result)

negative_samples = df[df['output'] == 'Negative']
positive_samples = df[df['output'] == 'Positive'].sample(n=len(negative_samples))
trainset = pd.concat([negative_samples, positive_samples])

# 나머지 데이터를 testset으로 설정
testset = df.drop(trainset.index)

print(f"Trainset size: {len(trainset)}, Testset size: {len(testset)}")

# trainset을 JSON 파일로 저장
with open('data/title_classifier_trainset.json', 'w') as f:
    json.dump(trainset.to_dict(orient='records'), f, indent=2, ensure_ascii=False)

# testset을 JSON 파일로 저장
with open('data/title_classifier_testset.json', 'w') as f:
    json.dump(testset.to_dict(orient='records'), f, indent=2, ensure_ascii=False)