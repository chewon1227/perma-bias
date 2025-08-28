import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

with open('total_jobs.txt', 'r', encoding='utf-8') as f:
    jobs = f.readlines()
    jobs = [job.strip() for job in jobs]

prompt = """You will be given a single job title in various languages as input. Your task is to assign it to the most appropriate standardized cluster based on its semantic meaning and job taxonomy. The clustering should align with a specificity level similar to O*NET classifications, unifying variations of the same concept.

Examples:
Input: 'University Lecturer', Output: 'Professor'
Input: 'Backend Engineer', Output: 'Software Developer'
Input: 'Chef de Cuisine', Output: 'Chef'

Ensure the output is limited strictly to the clustered job title in English, consisting of 1 or 2 words. Do not include any additional information or context.

Job Name: """

clustered_jobs = {}

for job in tqdm(jobs):
    if job == '':
        continue
    response = gen(prompt + job)
    if len(response) > 80: # if the response is too long, try again
        response = gen(prompt + job)
    if len(response) > 80 or len(response) == 0: # if the response is still too long, or empty, mark as unknown
        response = 'Unknown'
    if ':' in response: # if the response contains the prompt, remove it
        response = response.split(':')[-1].strip()
    # print(job, '->', response)
    if response not in clustered_jobs:
        clustered_jobs[response] = []
    clustered_jobs[response].append(job)

print('Clustered jobs:', len(clustered_jobs))

with open('clustered_jobs.json', 'w', encoding='utf-8') as f:
    json.dump(clustered_jobs, f, ensure_ascii=False, indent=2)
