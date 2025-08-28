import os
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm  

#link paste
data_path = 'linkedin_job_postings.csv'
df = pd.read_csv(data_path)

meta_data_keys = ['got_summary', 'got_ner', 'is_being_worked']  # 메타 데이터에 넣을 것들 

grouped_data = defaultdict(lambda: {
    "job_link": None,
    "companies": [],
    "job_locations": [],
    "meta_data": {}
})

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
    job_title = row['job_title']
    
    grouped_data[job_title]["job_link"] = row['job_link']
    
    if row['company'] not in grouped_data[job_title]["companies"]:
        grouped_data[job_title]["companies"].append(row['company'])
    
    if row['job_location'] not in grouped_data[job_title]["job_locations"]:
        grouped_data[job_title]["job_locations"].append(row['job_location'])
    
    # meta key
    for key in meta_data_keys:
        if key in df.columns and pd.notna(row[key]):
            grouped_data[job_title]["meta_data"][key] = row[key]

sorted_grouped_data = dict(sorted(grouped_data.items()))

#saving
output_file = 'curated_job_data.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sorted_grouped_data, f, ensure_ascii=False, indent=4)

print(f"Finish: {output_file}")
