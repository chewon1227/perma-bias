from typing import List, Tuple, Dict
from collections import defaultdict
from careerpathway.scoring import load_diversity, load_issue
from careerpathway.utils import parse_jobs_and_skills, parse_jobs_and_skills_ko
from mylmeval.utils import open_json, save_json
import fire
import os
from treebase import Tree
from careerpathway.multiprompt import PROMPTS
from functools import partial


def _make_save_dir(top_k_list, method, language, data_type, do_bias, beam_size, prefix='results/eval_prompt3'):
    how = '_'.join([str(top_k)+method for top_k, method in zip(top_k_list, method)]) + f"_{beam_size}"
    if data_type == 'issue':
        return f"{prefix}_issue_{how}"
    if language != 'en':
        return os.path.join(f"{prefix}_{how}", language)
    if do_bias:
        return f"{prefix}_{do_bias}_{how}"


def gentree(
    model_name_or_path: str = 'Qwen/Qwen2.5-3B-Instruct',
    start: int | None = None,
    batch_size: int = 64,
    method: List[str] = ['gen','gen','gen'], # h3
    top_k_list : List | str | Tuple = ["30", "2", "2", "2"],
    language: str = 'en',
    temperature: float = 0.7,
    beam_size: int = None,
    do_bias: bool = False, # us_F, us_M
    save_tmp: bool = False,
    data_type: str = 'diversity' # 'diversity' or 'issue'
    ):
    
    if isinstance(top_k_list, str):
        top_k_list = [int(r.strip("")) for r in top_k_list.split(",")]
    if len(top_k_list) != len(method):
        method = ['gen'] * len(top_k_list)
        
    parsing_function_with_country = partial(parse_jobs_and_skills, country_code=language)
    save_dir = _make_save_dir(top_k_list, method, language, data_type, do_bias, beam_size)
    print(f"save_dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    if data_type == 'issue':
        queries, graphs = load_issue(test_size=50, graph_version=False, only_en=True)
    else:
        queries, graphs = load_diversity(test_size=50)
        
    pipeline = Tree(
        batch_size=batch_size, 
        model_name_or_path=model_name_or_path, 
        top_k_list=top_k_list,
        method=method,
        language=language.split("_")[0],
        save_dir=save_dir,
        prompt=PROMPTS[language],
        parsing_function=parsing_function_with_country,
        temperature=temperature,
        beam_size=beam_size,
        do_bias=bool(do_bias),
        save_tmp=save_tmp
        )

    trees = pipeline.run(
        queries=queries[start:start+10] if start is not None else queries,
        load_where='results/eval_diversity_1/{}'
    )

    save_json(trees, f'{save_dir}/{model_name_or_path.replace("/", "_")}{f"_{start}" if start != None else ""}.jsonl')
    

def rettree(
    model_name_or_path: str = 'Qwen/Qwen2.5-3B-Instruct',
    start: int | None = None,
    batch_size: int = 64,
    method: List[str] = ['ret', 'ret', 'ret'], # h3
    top_k_list : List | str | Tuple = "10,2,2",
    language: str = 'en'
    ):
    
    if isinstance(top_k_list, str):
        top_k_list = [int(r.strip("")) for r in top_k_list.split(",")]
    elif isinstance(top_k_list, Tuple):
        top_k_list = list(top_k_list)
    if len(top_k_list) != len(method):
        method = ['ret'] * len(top_k_list)  
    save_dir = f"results/ret_{'_'.join([str(top_k)+method for top_k, method in zip(top_k_list, method)])}"
    if language != 'en':
        save_dir = save_dir.replace('eval_', f'{language}_eval_')
    
    queries, graphs = load_diversity(test_size=50)
    queries = queries[:5]
    
    pipeline = Tree(
        batch_size=batch_size, 
        model_name_or_path=model_name_or_path, 
        top_k_list=top_k_list,
        method=method,
        language=language,
        save_dir=save_dir,
        )

    trees = pipeline.run(
        queries=queries,
        load_where='results/eval_diversity_1/{}'
    )

    os.makedirs(save_dir, exist_ok=True)
    save_json(trees, f'{save_dir}/{model_name_or_path.replace("/", "_")}{f"_{start}" if start != None else ""}.jsonl')
      

def rettreegen(
    model_name_or_path: str = 'Qwen/Qwen2.5-3B-Instruct',
    start: int | None = None,
    batch_size: int = 64,
    method: List[str] | str = ['pivot_gen', 'pivot_gen', 'pivot_gen'], # h3
    top_k_list : List | str | Tuple = "100,10,10",
    language: str = 'en',
    temperature: float = 0.8
):
    if isinstance(top_k_list, str):
        top_k_list = [int(r.strip("")) for r in top_k_list.split(",")]
    elif isinstance(top_k_list, Tuple):
        top_k_list = list(top_k_list)
        3
    save_dir = f"results/retgen_{'_'.join([str(top_k)+method for top_k, method in zip(top_k_list, method)])}"
    if language != 'en':
        save_dir = save_dir.replace('eval_', f'{language}_eval_')
    
    queries, graphs = load_diversity(test_size=50)
    queries = queries[:5]
    
    pipeline = Tree(
        batch_size=batch_size, 
        model_name_or_path=model_name_or_path, 
        top_k_list=top_k_list,
        method=method,
        language=language,
        save_dir=save_dir,
        prompt=PROMPTS[language],
        parsing_function=parse_jobs_and_skills if language == 'en' else parse_jobs_and_skills_ko,
        temperature=temperature
        )

    trees = pipeline.run(
        queries=queries,
        load_where='results/eval_diversity_1/{}'
    )

    os.makedirs(save_dir, exist_ok=True)
    save_json(trees, f'{save_dir}/{model_name_or_path.replace("/", "_")}{f"_{start}" if start != None else ""}.jsonl')
    

if __name__ == "__main__":
    fire.Fire({
        'gentree': gentree,
        'rettree': rettree,
        'rettreegen': rettreegen
    })