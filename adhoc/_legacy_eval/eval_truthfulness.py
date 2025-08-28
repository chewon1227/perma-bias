from mylmeval import save_json, open_json, get_results
import fire
import os
import random
from typing import Dict, List

def get_oneshot_prompt(task: str, language: str, examples: List[Dict] = None) -> str:
    language_option = {
        'ja': ['JPY', 'Japanese'],
        'en': ['USD', 'English'],
        'ko': ['KRW', 'Korean'],
        'es': ['EUR', 'Spanish'],
    }
    
    template_map = {
        'salary': {
            'prefix': "Provide only the median salary OR the 25th-75th percentile salary range for the given job position in given unit, per year.\n\nExamples:\n",
            'format': "Job: {}\nSalary: {}\n\n"
        },
        'requirement': {
            'prefix': f"Given a job position, provide the minimum qualification requirements in {language_option[language][1]}.\n\nExamples:\n",
            'format': "Job: {}\nRequirements: {}\n\n"
        },
        'description': {
            'prefix': f"Given a job position, provide a brief description in {language_option[language][1]}.\n\nExamples:\n",
            'format': "Job: {}\nDescription: {}\n\n"
        }
    }
    
    template = template_map[task]
    prompt = template['prefix']
    
    # Add examples if provided
    if examples:
        for example in examples:
            prompt += template['format'].format(
                example['input'],
                example['groundtruth']
            )
    
    # Add the actual query
    prompt += "Now, please provide information for the following job:\nJob: {}"
    
    return prompt

def get_data(data_path: str = 'data/evalset/truthfulness.jsonl', shots: int = 1) -> List[Dict]:
    """
    Process evaluation data using examples from the same dataset.
    
    Args:
        data_path (str): Path to the evaluation dataset
        shots (int): Number of examples to include in the prompt
        
    Returns:
        List[Dict]: Processed evaluation dataset with examples
    """
    data = open_json(data_path)
    evalset = []
    
    for idx, d in enumerate(data):
        # Get examples from other data points with the same task and language
        possible_examples = [
            x for i, x in enumerate(data) 
            if i != idx  # Exclude current data point
            and x['task'] == d['task']  # Same task
            and x['language'] == d['language']  # Same language
        ]
        
        # Randomly select examples if we have enough
        if possible_examples and shots > 0:
            selected_examples = random.sample(possible_examples, min(shots, len(possible_examples)))
        else:
            selected_examples = []
        
        # Generate prompt with selected examples
        prompt = get_oneshot_prompt(d['task'], d['language'], selected_examples)
        
        evalset.append({
            'inputs': [prompt.format(d['input'])],
            'groundtruth': d['groundtruth'],
            'metadata': {k:v for k,v in d.items() if k not in ['input', 'groundtruth']}
        })
    
    return evalset


def infer(
    model_name_or_path: str = 'mistralai/Misral-8B-Instruct-v0.3',
    shots: int = 1
    ):
    
    os.makedirs('results/eval_truthfulness', exist_ok=True)
    data = get_data(shots=shots)
    get_results(
        model_name_or_path=model_name_or_path,
        prompt="{}",
        data=data,
        max_tokens=256,
        do_log=True,
        batch_size=len(data) if 'gpt' not in model_name_or_path else 1,
        apply_chat_template='auto',
        while_loop=True if 'gemini' in model_name_or_path else False,
        save_path='results/eval_truthfulness/{}.jsonl'.format(model_name_or_path.replace('/', '_'))
    )


if __name__ == '__main__':
    fire.Fire(infer)