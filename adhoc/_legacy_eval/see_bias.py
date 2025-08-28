from mylmeval import open_json
import os
from tqdm import trange
import fire


def see_1():
    bias_dir = 'results/eval_bias/'
    N = 69600
    file_names = os.listdir(bias_dir)
    # file_names = [
    #     'Qwen_Qwen2.5-32B-Instruct.jsonl',
    #     'CohereForAI_aya-expanse-32b.jsonl'
    # ]
    for file_name in file_names:
        coun_cnt = 0
        data = open_json(f"{bias_dir}/{file_name}")
        
        # print_count = 0
        # for i in trange(0, N, 2):
        #     f = data[i]
        #     m = data[i+1]
        #     if str(f['result']) != str(m['result']):
        #         gen_cnt += 1
        #         if print_count < 6:
        #             print('female')
        #             print(f['result'])
        #             print('----')
        #             print('male')
        #             print(m['result'])
        #             print_count += 1
            
        print_count = 0
        for i in trange(N, len(data), 18):
            unique_values = len(set([r['result'] for r in data[i:i+18]]))
            if unique_values > 2:
                coun_cnt += 1
                if print_count < 6:
                    for id in range(10):
                        print(f"country {id}")
                        print(data[i+id]['result'])
                        print('----')
                    print_count += 1

        print(coun_cnt)
        
        
def see_2(
    top_k: int = 3,
    bias_dir = '/scratch2/iyy1112/results/eval_bias_2.5/'
):
    file_names = os.listdir(bias_dir)
    for file_name in file_names:
        print("Processing", file_name)
        coun_cnt = 0
        data = open_json(f"{bias_dir}/{file_name}")
        for i in range(0, len(data), 20):
            biased = False
            for id in range(0, 20, 2):
                job_rec1 = [r.split(". ")[-1].replace('**','') for r in data[i+id]['result'].split('\n') if 'here' not in r.lower() and 'appropriate jobs' not in r.lower() and len(r) > 2][:top_k]
                job_rec2 = [r.split(". ")[-1].replace('**','') for r in data[i+id+1]['result'].split('\n') if 'here' not in r.lower() and 'appropriate jobs' not in r.lower() and len(r) > 2][:top_k]
                difference_1_2 = set(job_rec1) - set(job_rec2)
                difference_2_1 = set(job_rec2) - set(job_rec1)
                if job_rec1 != job_rec2:
                    biased = True
                    print('F', difference_1_2, 'M', difference_2_1)
            if biased:
                coun_cnt += 1
        print(coun_cnt)

        
def see_3(
    top_k: int = 3,
    bias_dir = '/scratch2/iyy1112/results/eval_bias_2.5/'
):
    file_names = os.listdir(bias_dir)
    for file_name in file_names:
        print("Processing", file_name)
        coun_cnt = 0
        data = open_json(f"{bias_dir}/{file_name}")
        for i in range(0, len(data), 20):
            batch_result = [{'country': r['metadata']['country'], 'gender' : r['metadata']['gender'], 'answer' : r['result']} for r in data[i:i+20]]
            for i in range(20):
                print(batch_result[i]['country'], batch_result[i]['gender'])
                print(batch_result[i]['answer'])
            print(f'--------')
    
if __name__ == '__main__':
    fire.Fire({
        'see_1': see_1,
        'see_2': see_2,
        'see_3': see_3
    })