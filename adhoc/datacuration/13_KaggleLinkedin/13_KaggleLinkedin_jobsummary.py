import pandas as pd
import json
from tqdm import tqdm  
import re

def extract_job_from_link(link)
    match = re.search(r'https://www\.linkedin\.com/jobs/view/([-\w]+)', link)
    if match:
        job_name = match.group(1).replace('-', ' ')
        return job_name
    return None

def process_csv(file_path):
    df = pd.read_csv(file_path)
    job_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        job_link = row['job_link']
        job_summary = row['job_summary']
        
        job_name = extract_job_from_link(job_link)
        
        if job_name:
            job_data.append({
                'job': job_name,
                'job_summary': job_summary
            })

    return job_data

def save_as_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # file path 
    csv_file_path = 'job_summary.csv'  
    output_json_path = 'data/job_summaries_output.json'

    job_data = process_csv(csv_file_path)

    save_as_json(job_data, output_json_path)

    print(f"Finish: {output_json_path}")
    print(f"# of worked data: {len(job_data)}")
