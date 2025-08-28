import pandas as pd
import json
from tqdm import tqdm  

def process_csv(file_path):
    df = pd.read_csv(file_path)
    job_data = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        job_title = row['title']
        company_name = row['company_name']
        job_description = row['description']
        url = row['job_posting_url']
        
        max_salary = row.get('max_salary', None)
        min_salary = row.get('min_salary', None)
        salary = f"{min_salary} - {max_salary}" if pd.notna(min_salary) and pd.notna(max_salary) else None
        
        additional_info = {
            'location': row.get('location', None),
            'employment_type': row.get('employment_type', None),
            'experience_level': row.get('experience_level', None)
        }

        job_data.append({
            'job_title': job_title,
            'company_name': company_name,
            'job_description': job_description,
            'url': url,
            'salary': salary,
            'additional_info': additional_info
        })

    return job_data

def save_as_json(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # file path
    csv_file_path = '/Users/chaewon/Desktop/MIR_LAB/postings.csv'  
    output_json_path = 'data/job_postings_output.json'

    job_data = process_csv(csv_file_path)

    save_as_json(job_data, output_json_path)

    print(f"Finish: {output_json_path}")
    print(f"# of worked data: {len(job_data)}")
