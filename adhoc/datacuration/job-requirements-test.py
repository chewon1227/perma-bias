import json
from tqdm import tqdm
import sys
import similarity

model_name = sys.argv[1]

with open(f"linkedin_requirements_test_data.jsonl",'r',encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(0)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def gen(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

evaluator = similarity.Similarity(batch_size=1)
predictions = []
references = []

for d in tqdm(dataset[:10]):
    response = gen(d['messages'][0]['content'])
    gt = d['messages'][1]['content']
    # print('Response:', response)
    # print('GT:', gt)
    predictions.append(response)
    references.append(gt)

scores = evaluator.evaluate(predictions, references, return_all=False)

model_name = model_name.split('/')[-1]

# with open(f'job-requirements-test-scores-{model_name}.txt','w') as f:
#     for key, value in scores.items():
#         f.write(f"{key}: {value}\n")

scores_10 = evaluator.evaluate(predictions, references, return_all=True)

outputs = [
    {
        'prediction': predictions[i],
        'reference': references[i],
        'scores': {
            'bleu': float(scores_10['bleu'][i]),
            'rouge_1': float(scores_10['rouge_1'][i]),
            'rouge_2': float(scores_10['rouge_2'][i]),
            'rouge_l': float(scores_10['rouge_l'][i]),
            'meteor': float(scores_10['meteor'][i]),
            'bert_score': float(scores_10['bert_score'][i]),
        }
    }
    for i in range(10)
]

with open(f'job-requirements-test-scores-{model_name}-10.json','w') as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)