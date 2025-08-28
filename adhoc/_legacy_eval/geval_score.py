from mylmeval import save_json, open_json, get_results
import fire
import os

PROMPT = """Please mark the similarity score from 1 to 5 for the each responses and groundtruth response, without extra words.
The score 1 means the response is not similar to the groundtruth response, and the score 5 means the response is very similar to the groundtruth response.


[Groundtruth]: {}
{}

[Evaluation]:"""

def geval(
    model_name_or_path: str = 'gpt-4o-mini',
    task: str = 'truthfulness-description',
):
    task, subtask = task.split('-')
    models = [r for r in os.listdir(f'results/eval_{task}') if '70B' not in r and 'geval' not in r]
    datas = dict()
    for model in models:
        model_data = open_json(f'results/eval_{task}/{model}')
        print(model, len(model_data))
        datas[model.replace("_", "/")] = model_data
        
    real_input = []
    for i in range(len(model_data)):
        if model_data[i]['metadata']['type'] != 'salary':
            real_input.append({
                'inputs' : [
                    model_data[i]['groundtruth'],
                    "\n".join([f"{id+1}. {v[i]['result']}" for id, (k, v) in enumerate(datas.items())])
                    ],
                'metadata': {'idx': i, 'models' : list(datas.keys())}
                })
        
    
    get_results(
        model_name_or_path=model_name_or_path,
        data=real_input,
        prompt=PROMPT,
        max_tokens=50,
        batch_size=1,
        apply_chat_template='auto',
        save_path=f'results/eval_{task}/geval_{subtask}.jsonl',
    )
          




if __name__ == "__main__":
    import pdb ; pdb.set_trace()
    
    fire.Fire(geval)
    