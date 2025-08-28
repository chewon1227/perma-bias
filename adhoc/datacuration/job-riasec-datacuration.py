import csv

job_list = []

def get_csv_data(riasec_name):
    csv_data = []
    with open(f'data/{riasec_name}.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            csv_data.append(row)
    return csv_data[1:]

job_list += get_csv_data('Realistic')
job_list += get_csv_data('Investigative')
job_list += get_csv_data('Artistic')
job_list += get_csv_data('Social')
job_list += get_csv_data('Enterprising')
job_list += get_csv_data('Conventional')

seen = set()
unique_job_list = []
for row in job_list:
    if row[2] not in seen:
        unique_job_list.append(row)
        seen.add(row[2])

def is_riasec_in_job(riasec, job):
    if riasec == job[0][0]:
        return 1 # first riasec code
    elif riasec in job[0]:
        return -1 # 2nd or 3rd riasec code
    else:
        return 0 # negative

riasec_job_list = []
for job in unique_job_list:
    # RIASEC 코드에 맞춰 각각 포함되면 1, 아니면 0으로 job 정보를 리스트에 추가
    riasec_job_list.append([job[3], 'R', is_riasec_in_job('R', job)])
    riasec_job_list.append([job[3], 'I', is_riasec_in_job('I', job)])
    riasec_job_list.append([job[3], 'A', is_riasec_in_job('A', job)])
    riasec_job_list.append([job[3], 'S', is_riasec_in_job('S', job)])
    riasec_job_list.append([job[3], 'E', is_riasec_in_job('E', job)])
    riasec_job_list.append([job[3], 'C', is_riasec_in_job('C', job)])

# -1인 줄만 남기기
riasec_2nd_3rd_job_list = [job for job in riasec_job_list if job[2] == -1]

# -1인 줄은 제거
riasec_job_list = [job for job in riasec_job_list if job[2] != -1]

# 1 개수 세기
positive_count = 0
for job in riasec_job_list:
    if job[2] == 1:
        positive_count += 1

import json

with open('data/riasec-job-data.json','w',encoding='utf-8') as f:
    json.dump(riasec_job_list,f,indent=2,ensure_ascii=False)

with open('data/riasec-augmented-job-data.json','r',encoding='utf-8') as f:
    json_data = json.load(f)

# 1 < n <= 10
def get_top_n_and_m_sentences(riasec, n, m):
    sentences = []
    for data in json_data:
        if data[2] == riasec:
            sentences.append(data)
    seen_gt_sentences_count = {}
    top_n_sentences = []
    top_m_sentences = []
    for sentence in sentences:
        if sentence[1] not in seen_gt_sentences_count:
            seen_gt_sentences_count[sentence[1]] = 1
            top_n_sentences.append([sentence[1], sentence[1], riasec])
            top_m_sentences.append([sentence[1], sentence[1], riasec])
        if seen_gt_sentences_count[sentence[1]] < n:
            top_n_sentences.append(sentence)
        elif seen_gt_sentences_count[sentence[1]] < n + m - 1:
            top_m_sentences.append(sentence)
        seen_gt_sentences_count[sentence[1]] += 1
    return top_n_sentences, top_m_sentences

pos_aug_sentences = []
neg_aug_sentences = []
pos_sentences, neg_sentences = get_top_n_and_m_sentences('R', 7, 2)
pos_aug_sentences += pos_sentences
neg_aug_sentences += neg_sentences
pos_sentences, neg_sentences = get_top_n_and_m_sentences('I', 7, 2)
pos_aug_sentences += pos_sentences
neg_aug_sentences += neg_sentences
pos_sentences, neg_sentences = get_top_n_and_m_sentences('A', 7, 2)
pos_aug_sentences += pos_sentences
neg_aug_sentences += neg_sentences
pos_sentences, neg_sentences = get_top_n_and_m_sentences('S', 7, 2)
pos_aug_sentences += pos_sentences
neg_aug_sentences += neg_sentences
pos_sentences, neg_sentences = get_top_n_and_m_sentences('E', 7, 2)
pos_aug_sentences += pos_sentences
neg_aug_sentences += neg_sentences
pos_sentences, neg_sentences = get_top_n_and_m_sentences('C', 7, 2)
pos_aug_sentences += pos_sentences
neg_aug_sentences += neg_sentences

pos_aug_sentences_dict = {'R':[], 'I':[], 'A':[], 'S':[], 'E':[], 'C':[]}
neg_aug_sentences_dict = {'R':[], 'I':[], 'A':[], 'S':[], 'E':[], 'C':[]}
for sentence in pos_aug_sentences:
    pos_aug_sentences_dict[sentence[2]].append(sentence)
for sentence in neg_aug_sentences:
    neg_aug_sentences_dict[sentence[2]].append(sentence)

data=[]
for job in riasec_job_list:
    append_data = []
    if job[2] == 1:
        for sentence in pos_aug_sentences_dict[job[1]]:
            append_data.append({'job': job[0],
                                'riasec': job[1],
                                'positive': job[2],
                                'statement': sentence[0],
                                'original_statement': sentence[1]})
    else:
        for sentence in neg_aug_sentences_dict[job[1]]:
            append_data.append({'job': job[0],
                                'riasec': job[1],
                                'positive': job[2],
                                'statement': sentence[0],
                                'original_statement': sentence[1]})
    data += append_data

positive_count = 0
for job in data:
    if job['positive'] == 1:
        positive_count += 1

print("positive:", positive_count)
print("negative:",len(data)-positive_count)

with open('data/riasec-augmented-final-data.json','w',encoding='utf-8') as f:
    json.dump(data,f,indent=2,ensure_ascii=False)

import random
random.seed(0)
random.shuffle(data)

llm_data_0_or_1 = []
for job in data:
    llm_data_0_or_1.append({'messages':
                            [{'role': 'user', 'content': f"Answer 1 or 0 for the fitness of the given user information and job name.\nUser information: {job['statement']}\nJob name: {job['job']}"},
                             {'role': 'assistant', 'content': f"{job['positive']}"}]})

training_data_0_or_1 = llm_data_0_or_1[:-900]
test_data_0_or_1 = llm_data_0_or_1[-900:]

llm_data_pos_neg = []
for job in data:
    llm_data_pos_neg.append({'messages':
                             [{'role': 'user', 'content': f"Answer positive or negative for the fitness of the given user information and job name.\nUser information: {job['statement']}\nJob name: {job['job']}"},
                              {'role': 'assistant', 'content': f'{"positive" if job["positive"] == 1 else "negative"}'}]})
    
training_data_pos_neg = llm_data_pos_neg[:-900]
test_data_pos_neg = llm_data_pos_neg[-900:]

with open('data/riasec-training-data-0-or-1.jsonl','w',encoding='utf-8') as f:
    for d in training_data_0_or_1:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data/riasec-test-data-0-or-1.jsonl','w',encoding='utf-8') as f:
    for d in test_data_0_or_1:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data/riasec-training-data-pos-neg.jsonl','w',encoding='utf-8') as f:
    for d in training_data_pos_neg:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

with open('data/riasec-test-data-pos-neg.jsonl','w',encoding='utf-8') as f:
    for d in test_data_pos_neg:
        f.write(json.dumps(d,ensure_ascii=False)+'\n')

riasec_2nd_3rd_data = []
for job in riasec_2nd_3rd_job_list:
    seen_sentence = set()
    for sentence in json_data:
        if sentence[2] == job[1] and sentence[1] not in seen_sentence:
            seen_sentence.add(sentence[1])
            riasec_2nd_3rd_data.append([job[0], job[1], sentence[1]])

with open('data/riasec-2nd-3rd-data.json','w',encoding='utf-8') as f:
    json.dump(riasec_2nd_3rd_data,f,indent=2,ensure_ascii=False)
    