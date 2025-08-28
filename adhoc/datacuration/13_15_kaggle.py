import json

data1_path = 'data13_KaggleLinkedin_merged.json'
data2_path = 'data15_jobpostings_output.json'
result = []

data1 = json.load(open(data1_path))
data2 = json.load(open(data2_path))

for da in data1:
    result.append({
        'jobname' : data['job_name'],
        'jobskills' : data['jobskills'],
        'description' : data['jobsummary'],
        'metadata' : {k:v for k, v in da.items() if k not in ['job_name', 'jobskills', 'jobsummary']},
    })
    
for da in data2:
    result.append({
        'jobname' : data['job_name'],
        'skill' : None,
        'description' : data['job_description'],
        'metadata' : {'job_link': data['job_link'], 'company': data['company'], 'salary': data['salary'], **data['additional_info']},
    })
    
# save as jsonl
with open('data13_15_kaggle.jsonl', 'w') as f:
    for da in result:
        f.write(json.dumps(da) + '\n')