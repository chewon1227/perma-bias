from mylmeval.utils import save_json, open_json
from mylmeval.infer import get_results
import fire
from collections import defaultdict
import os
import random
from careerpathway.scoring import load_diversity, load_high_qual_diversity
from careerpathway.utils import get_random_name
from careerpathway.scoring import Diversity

def main(
    model_name_or_path: str = 'Qwen/Qwen2.5-0.5B-Instruct',
    start: int = None,
    option_num: int = 10,
    language: str = 'us',
    depth: int | None = 5,
    bias: str = None, # us_F
    only_parsing: bool = False
    ):
    
    if only_parsing:
        keyword = [
            'psychology',
            'business',
            'engineering',
            'medicine',
            'law',
            'education',
            'arts',
            'entertainment',
            'health',
            'social',
            'electrical',
            'computer',
            'math',
            'physical',
        ]
        diversity = Diversity()
        for keyword in keyword:
            testset, _ = load_high_qual_diversity(test_size=10000, do_keyword=True, keyword=keyword, only_main=True)
            career_dots = [item['nodes'] for item in testset]
            career_dots = [item for sublist in career_dots for item in sublist]
            print(f"keyword: {keyword}, {len(career_dots)}")
            results = diversity.evaluate([career_dots])
        
        return

    if bias == 'None':
        bias = None
    save_dir = 'results/eval_diversity_psy' if option_num == 10 else f'results/eval_diversity_{option_num}'
    save_dir = save_dir.replace('eval_diversity', f'eval_diversity_{depth}') if depth != None else save_dir
    save_dir = save_dir.replace('eval_diversity', f'eval_diversity_{bias}') if bias != None else save_dir
    save_dir = os.path.join(save_dir, language) if language != 'us' else save_dir
    
    os.makedirs(f'{save_dir}', exist_ok=True)
    
    prompt_template = """Given a person in the position of {}, 
what are the {} most recommendable career steps? 
List only the job titles, one per line."""

    if bias:
        random_name = get_random_name(bias.split("_")[0], bias.split("_")[1]) # nation, sex
        prompt_template = prompt_template.replace('sequential steps.\n', f'sequential steps.\nThe person with this position has this name: {random_name}\n')
        
    testset, _ = load_high_qual_diversity(test_size=250, do_keyword=True, keyword='psychology')
    data = [{
        'inputs' : [item['initial_node'], option_num],
        'meta' : {'graph_id' : item['graph_id']}, 
    } for item in testset]
        
        
    results = get_results(
        model_name_or_path=model_name_or_path,
        data=data[start:] if start != None else data,
        prompt=prompt_template,
        max_tokens=8000,
        batch_size=len(data),
        apply_chat_template='auto',
        save_path=f'{save_dir}/{model_name_or_path.replace("/", "_")}{f"_{start}" if start != None else ""}.jsonl'
    )


if __name__ == "__main__":
    fire.Fire(main)