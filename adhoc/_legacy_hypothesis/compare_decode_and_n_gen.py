import os
import fire 
from mylmeval.llm import MyLLMEval
from mylmeval.utils import open_json
from careerpathway.scoring.load_testset import load_high_qual_diversity
from careerpathway.scoring.diversity import Diversity


def compare(
    model_name_or_path: str = 'Qwen/Qwen2.5-3B-Instruct',
    sample_n : int = 10,
    temperature : float = 0.7,
    only_parsing: bool = False,
    top_k: int = 50,
    top_p: float = 0.9
):
    myllmeval = MyLLMEval(model_name_or_path)
    testset, graphs = load_high_qual_diversity(test_size=250, do_keyword=True) # 'Psychology'
    data = [{**item, 'inputs': [item['initial_node']]} for item in testset]
    os.makedirs(f'results/eval_decond_n_gen', exist_ok=True)
    
    decoding_save_path = f'results/{nowdate}_decond_n_gen/decoding_{model_name_or_path.replace("/", "_")}_{sample_n}_{temperature}.jsonl'
    n_generation_save_path = f'results/{nowdate}_decond_n_gen/n_generation_{model_name_or_path.replace("/", "_")}_{sample_n}_{temperature}.jsonl'
    diversity = Diversity()
    
    if only_parsing:
        decoding_results = open_json(decoding_save_path)
        n_generation_results = open_json(n_generation_save_path)
        
        decoding_diversity = diversity.evaluate([item['result'] for item in decoding_results])
        n_generation_diversity = diversity.evaluate([item['result'].split("\n") for item in n_generation_results])
    
        return
    
    if os.path.exists(decoding_save_path):
        os.remove(decoding_save_path)
    if os.path.exists(n_generation_save_path):
        os.remove(n_generation_save_path)
        
    # sampling with decoding
    prompt = "Please suggest an appropriate job for this careerpath:\n\nCareer: {}\n\nJust suggest a job without any other information."
    _ = myllmeval.inference(
        prompt=prompt,
        data=data,
        n_generations=sample_n,
        temperature=temperature,
        save_path=decoding_save_path,
        top_k=top_k,
        top_p=top_p
    )
    
    # n generation
    prompt = f"Please suggest {sample_n} appropriate jobs for this careerpath:\n\n"+"Career: {}\n\nJust suggest jobs without any other information."
    _ = myllmeval.inference(
        prompt=prompt,
        data=data,
        n_generations=1,
        temperature=temperature,
        save_path=n_generation_save_path,
        top_k=top_k,
        top_p=top_p
    )

    decoding_results = open_json(decoding_save_path)
    n_generation_results = open_json(n_generation_save_path)
    
    decoding_diversity = diversity.evaluate([item['result'] for item in decoding_results])
    n_generation_diversity = diversity.evaluate([item['result'].split("\n") for item in n_generation_results])
    
    return
        

if __name__ == '__main__':
    fire.Fire(compare)