from careerpathway.utils import open_json
import random
import json
import os
from termcolor import colored

def load_reddit_data(sampling_num : int = None, seed : int = 42):
    data = open_json('data/data3_reddit.json')
    data = [{'inputs' : [item['Title'], item['Post Text']]} for item in data]
    if sampling_num is not None:
        random.seed(seed)
        data = random.sample(data, sampling_num)
    return data


def load_nemotron_persona(output_path: str = 'data/nemotron_personas.json', sampling_num: int = 10000, seed: int = 42):
    if os.path.exists(output_path):
        data_to_save = open_json(output_path)
    else:
        from datasets import load_dataset
        nemotron_personas = load_dataset("nvidia/Nemotron-Personas", "default")
        data_to_save = [
            {'name': item['persona'].split(",")[0], **item} for item in nemotron_personas['train']]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    
    # Apply sampling regardless of whether data was loaded from file or downloaded
    if sampling_num is not None and len(data_to_save) > sampling_num:
        random.seed(seed)
        data_to_save = random.sample(data_to_save, sampling_num)
        
    return data_to_save


def inspect_data(data_path: str, columns: list[str] = ['prompt', 'result'], random_n: int = 10, do_print: bool = True):
    data = open_json(data_path)
    subsample = random.sample(data, random_n)
    for item in subsample:
        for column in columns:
            if do_print:
                print(colored(column, 'green'), f"{item[column]}")
        if do_print:
            print('-'*100)
    return subsample