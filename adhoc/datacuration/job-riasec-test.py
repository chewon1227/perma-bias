import json
from tqdm import tqdm
import sys

model_name = sys.argv[1]
if 'pos' in model_name:
    BINARY_OR_POSITIVE = True
else:
    BINARY_OR_POSITIVE = False

with open(f"riasec-test-data-{'pos-neg' if BINARY_OR_POSITIVE else '0-or-1'}.jsonl",'r',encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

from transformers import AutoModelForCausalLM, AutoTokenizer

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
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

score=0

for d in tqdm(dataset):
    response = gen(d['messages'][0]['content'])
    gt = d['messages'][1]['content']
    if response == gt:
        score += 1

print(f"Accuracy: {score/len(dataset)}")

model_name = model_name.split('/')[-1]

with open(f'riasec-2nd-3rd-data.json', 'r', encoding='utf-8') as f:
    riasec_2nd_3rd_data = json.load(f)

score_2nd_3rd = 0

for d in tqdm(riasec_2nd_3rd_data):
    if BINARY_OR_POSITIVE:
        prompt = f"Answer positive or negative for the fitness of the given user information and job name.\nUser information: {d[2]}\nJob name: {d[0]}"
    else:
        prompt = f"Answer 1 or 0 for the fitness of the given user information and job name.\nUser information: {d[2]}\nJob name: {d[0]}"
    response = gen(prompt)
    gt_response = 'positive' if BINARY_OR_POSITIVE else '1'
    if response == gt_response:
        score_2nd_3rd += 1

print(f"2nd 3rd Accuracy: {score_2nd_3rd/len(riasec_2nd_3rd_data)}")

with open(f'riasec-test-scores-{model_name}.txt','w') as f:
    f.write(f'Score: {score/len(dataset)}\n')
    f.write(f'2nd 3rd Score: {score_2nd_3rd/len(riasec_2nd_3rd_data)}\n')