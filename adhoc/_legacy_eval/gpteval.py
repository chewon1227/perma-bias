from mylmeval.utils import open_json, save_json
from mylmeval.infer import get_results
import fire
import random
import pandas as pd
from collections import Counter
from careerpathway.utils import extract_num


def main(
    model_name_or_path: str = 'o1',
    data_path: str = 'data/humaneval/connectd3.json',
    only_parsing: bool = False
):
    
    def _parse(text):
        # split ',' or '\n'
        result = text.replace('\n', ',').split(',')
        
        if len(result) == 3 and len([r for r in result if r == 'A' or r == 'B']) == 3:
            return result
        else:
            result = [r.split(": ")[-1].split(". ")[-1].strip().strip("(").strip(")").strip(".") for r in result]
            result = [r for r in result if r == 'A' or r == 'B']
            if len(result) == 3:
                return result
            else:
                result = text.split("\n\m")
        return result
    
        
    if only_parsing:
        data = open_json(f'results/LLM-as-a-judge_{model_name_or_path}.json')
        answers = [_parse(r['result']) for r in data]
        answers = [item for sublist in answers for item in sublist]
        gt = [r['answer'] for r in data]
        
        s1 = sum([1 for a, g in zip(answers[0:len(answers):3], gt) if a == g])
        s2 = sum([1 for a, g in zip(answers[1:len(answers):3], gt) if a == g])
        s3 = sum([1 for a, g in zip(answers[2:len(answers):3], gt) if a == g])
        print(f"Option 1: {s1}/{len(gt)}")
        print(f"Option 2: {s2}/{len(gt)}")
        print(f"Option 3: {s3}/{len(gt)}")
        return
    
    add_prompt = """
    
Please select option for each metrics without any extra explanations (e.g., A, B, A)"""
    data = open_json(data_path)
    data = [{**item, 'inputs' : [item['prompt'] + add_prompt]} for item in data]  
    print(data[0]['inputs'][0])
    result = get_results(
        model_name_or_path=model_name_or_path,
        prompt="{}",
        data=data,
        max_tokens=500,
        batch_size=1,
        save_path=f'results/LLM-as-a-judge_{model_name_or_path}.json',
    )


if __name__ == '__main__':
    fire.Fire(main)