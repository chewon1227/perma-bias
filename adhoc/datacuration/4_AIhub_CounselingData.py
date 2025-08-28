import json
import pandas as pd
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def merge_data(data1, data2):
    merged_data = {}
    
    for idx, conversation in tqdm(data1.items(), desc="Merging Data"):
        student_idx = conversation['meta']['student_idx']
        
        if student_idx in data2:
            student_info = data2[student_idx]
            merged_data[student_idx] = {
                'meta': conversation['meta'],
                'conversation': conversation['conversation'],
                'student_info': student_info  
            }
        else:
            merged_data[student_idx] = {
                'meta': conversation['meta'],
                'conversation': conversation['conversation'],
                'student_info': {}
            }
    
    return merged_data

# json file path 
data1_path = '/counseling_element.json'
data2_path = 'info_element.json'

data1 = load_json(data1_path)
data2 = load_json(data2_path)

merged_data = merge_data(data1, data2)

output_path = 'data/output.json' #saving 
save_json(merged_data, output_path)
