from mylmeval import open_json, MyLLMEval
import os
import fire


def run(
    file_dir: str = 'results/eval_prompt3_10gen_2gen_2gen_2gen_0/Qwen_Qwen2.5-3B-Instruct_with_requirements.jsonl'
    ):
    if file_dir.endswith('.jsonl'):
        data = open_json(file_dir)
    else:
        data = open_json(os.path.join(file_dir, 'Qwen_Qwen2.5-3B-Instruct.jsonl'))
    myllmeval = MyLLMEval(model_path='Qwen/Qwen2.5-3B-Instruct')
    for item in data:
        nodes = item['nodes']
        for i in range(0, file_dir.count('gen')):
            print('Initial stage: {}'.format(item['initial_node']))
            stage_nodes = [n for n in nodes if len(n['parent_id']) == i]
            print(f"Stage {i}")
            print(', '.join([r['content'] for r in stage_nodes]))
            # for node in stage_nodes:
                # print(node)
                
        
        print()    
    print("Done")
    
    
if __name__ == "__main__":
    fire.Fire(run)