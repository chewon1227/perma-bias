import json
from tqdm import tqdm
import numpy as np
import re

with open('salary-test-data.jsonl','r',encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

with open('salary-test-data-with-per.jsonl','r',encoding='utf-8') as f:
    dataset_with_per = [json.loads(line) for line in f]

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(0)

def convert_to_year(salary, per):
    if per == 'year':
        return salary
    elif per == 'month':
        return salary * 12
    elif per == 'week':
        return salary * 52
    elif per == 'day':
        return salary * 5 * 52
    elif per == 'hour':
        return salary * 8 * 5 * 52

def output_to_salary(output, per=False):
    unit = output[:3] # KRW, USD, JPY, EUR
    output = output[4:]
    if output[-1] == '+':
        output = output[:-1]
    if per:
        # USD min 108000 - max 140000 per year
        per_idx = re.search(r' per ', output).start()
        per_name = output[per_idx+5:]
        output = output[:per_idx]
        if output[-1] == '+':
            output = output[:-1]
    if '-' in output:
        min_salary, max_salary = output.split('-')
        min_salary = min_salary.strip()
        max_salary = max_salary.strip()
        # remove min, max
        min_salary = int(min_salary[4:])
        max_salary = int(max_salary[4:])
        salary = (min_salary + max_salary) / 2
    else:
        salary = int(output)
    if not per:
        per_name = 'year'
    salary = convert_to_year(salary, per_name)
    print(salary, unit)
    return salary, unit

def test(model_name, per=False):
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
        print(response)
        return response

    krw_scores = []
    usd_scores = []
    jpy_scores = []
    # eur_scores = []
    all_usd_scores = []

    # https://www.irs.gov/individuals/international-taxpayers/yearly-average-currency-exchange-rates
    # 2023

    USD_TO_EUR_2023 = 0.924
    USD_TO_JPY_2023 = 140.511
    USD_TO_KRW_2023 = 1306.686

    if per:
        ds = dataset_with_per
    else:
        ds = dataset

    for d in tqdm(ds):
        response = gen(d['messages'][0]['content'])
        salary, unit = output_to_salary(response, per)
        gt_salary = d['messages'][1]['content']
        gt_salary, unit = output_to_salary(gt_salary, per)
        difference = abs(salary - gt_salary)
        if unit == 'KRW':
            krw_scores.append(difference)
            all_usd_scores.append(difference / USD_TO_KRW_2023)
        elif unit == 'USD':
            usd_scores.append(difference)
            all_usd_scores.append(difference)
        elif unit == 'JPY':
            jpy_scores.append(difference)
            all_usd_scores.append(difference / USD_TO_JPY_2023)
        # elif unit == 'EUR':
        #     eur_scores.append(difference)
        #     all_usd_scores.append(difference / USD_TO_EUR_2023)

    krw_scores = np.array(krw_scores)
    usd_scores = np.array(usd_scores)
    jpy_scores = np.array(jpy_scores)
    # eur_scores = np.array(eur_scores)
    all_usd_scores = np.array(all_usd_scores)

    return krw_scores, usd_scores, jpy_scores, all_usd_scores

model_list = [
    # '/scratch2/snail0822/career/job-salary-0.5b',
    '/scratch2/snail0822/career/job-salary-1.5b',
    '/scratch2/snail0822/career/job-salary-3b',
]

model_list_per = [
    # '/scratch2/snail0822/career/job-salary-0.5b-per',
    '/scratch2/snail0822/career/job-salary-1.5b-per',
    '/scratch2/snail0822/career/job-salary-3b-per',
]

for model_name in model_list:
    krw_scores, usd_scores, jpy_scores, all_usd_scores = test(model_name)
    # add to file
    model_name = model_name.split('/')[-1]
    with open(f'job-salary-test-scores.txt', 'a') as f:
        f.write(f'{model_name}\n')
        f.write(f'USD (all):\tmean: {all_usd_scores.mean()}\tstd: {all_usd_scores.std()}\n')
        f.write(f'KRW:\tmean: {krw_scores.mean()}\tstd: {krw_scores.std()}\n')
        f.write(f'USD:\tmean: {usd_scores.mean()}\tstd: {usd_scores.std()}\n')
        f.write(f'JPY:\tmean: {jpy_scores.mean()}\tstd: {jpy_scores.std()}\n')
        # f.write(f'EUR:\tmean: {eur_scores.mean()}\tstd: {eur_scores.std()}\n')
        f.write('\n')

for model_name in model_list_per:
    krw_scores, usd_scores, jpy_scores, all_usd_scores = test(model_name, per=True)
    # add to file
    model_name = model_name.split('/')[-1]
    with open(f'job-salary-test-scores.txt', 'a') as f:
        f.write(f'{model_name}\n')
        f.write(f'USD (all):\tmean: {all_usd_scores.mean()}\tstd: {all_usd_scores.std()}\n')
        f.write(f'KRW:\tmean: {krw_scores.mean()}\tstd: {krw_scores.std()}\n')
        f.write(f'USD:\tmean: {usd_scores.mean()}\tstd: {usd_scores.std()}\n')
        f.write(f'JPY:\tmean: {jpy_scores.mean()}\tstd: {jpy_scores.std()}\n')
        # f.write(f'EUR:\tmean: {eur_scores.mean()}\tstd: {eur_scores.std()}\n')
        f.write('\n')