import csv
import json

with open('data/jp_salary.csv','r',encoding='utf-8') as f:
    reader = csv.reader(f)
    data = [row for row in reader]

json_data = []
for row in data:
    json_data.append({
        "job": row[0],
        "job_category": row[1],
        "annual_salary": int(row[2])
    })

with open('data/jp_salary.json','w',encoding='utf-8') as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)
