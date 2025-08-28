import pandas as pd
import fire

def csv_to_labelstudio(
    data_path: str = 'urls.csv',
    sampling_num: int = 500,
    sub_lang: str = False,
    ) -> None:
    data = pd.read_csv(data_path, sep='\t', header=None)
    language = data_path.split('/')[-1].split("_")[0]
    try:
        data.columns = ['idx', 'url', 'title', 'job', 'language', 'query']
    except:
        data.columns = ['idx', 'url', 'title', 'job', 'language']
    if sub_lang:
        data = data[data['language'] == sub_lang]
        print(f'Filtering by {sub_lang} language, {len(data)} rows left')
    
    sub = data.sample(sampling_num, random_state=0)
    kor_dict = pd.read_json('/Users/sb/Downloads/workspace/Career-Pathway/data/data5_jobdict.json')
    kor_dict['ko'] = kor_dict['jobname'].apply(lambda x: x['ko'])
    kor_dict['en'] = kor_dict['jobname'].apply(lambda x: x['en'])
    kor_dict['sp'] = kor_dict['jobname'].apply(lambda x: x['sp'])
    kor_dict['ja'] = kor_dict['jobname'].apply(lambda x: x['ja'])

    sub = sub.merge(kor_dict, left_on='job', right_on=language, how='left')    
    sub[['idx', 'title', 'job', 'language', 'ko']].to_json(data_path.replace('.csv', '_s.json'), orient='records', indent=2, force_ascii=False)
    
    
if __name__ == '__main__':
    fire.Fire(csv_to_labelstudio)
     