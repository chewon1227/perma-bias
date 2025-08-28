import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from utils import save_json, open_json

def scrap_kor_jobdic():
    saved = []
    for i in range(1000, 10000):
        for j in range(1, 100):
            try:
                url = f'https://www.work.go.kr/consltJobCarpa/srch/jobDic/jobDicDtlInfo.do?pageType=initWord&jobCode={i}&jobSeq={j}&txtNumber=5&initWord=&stardWord=마&endWord=바&pageIndex=12'
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                job = soup.find('div', {'class': 'tit-job'}).text
                meaning = soup.find('div', {'class': "box-bluegray mb40"}).text.replace('\n', '').replace('\t', '')
                essentials = soup.select_one('#contents > div:nth-of-type(5) > div:nth-of-type(2)').get_text().replace('\n', '').replace('\t', '')
                infos = [r.get_text().replace('\n', '').replace('\t', '') for r in soup.select_one('#contents > div:nth-of-type(5)').find_all('li')]
                saved.append(
                    {
                        'code' : f"{i}-{j}",
                        'jobname' : job,
                        'definition' : meaning,
                        'explanation' : essentials,
                        'info' : infos
                    }
                )

                print(f"Success: {i}, {j}")

            except Exception as e:
                print(f"Error with {i}, {j}: {e}")
                break

        save_json(saved, 'data/job_dictionary.json')
        
def after_process():
    df = pd.read_json('/Users/sb/Downloads/workspace/Career-Pathway/data/job_dictionary.json')
    df['info'] = df['info'].apply(lambda x: {
        i.split(':')[0].strip(" ") : None if len(i.split(":")[-1].strip(" ")) < 2 else i.split(":")[-1].strip(" ") for i in x})
    
    df.to_json('/Users/sb/Downloads/workspace/Career-Pathway/data/kor_job_dictionary.json', orient='records', force_ascii=False, indent=2)
    

after_process()