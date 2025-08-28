import requests
import pandas as pd
from bs4 import BeautifulSoup
import tqdm
import json

codes = pd.read_excel('data/onet_codes.xlsx')

def crawl_onet():
    for code in tqdm.tqdm(codes.to_dict('records')):
        link = f'https://www.onetonline.org/link/summary/{code["idx"]}'
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        keys = [r.text for r in soup.find_all('h2', {'class' : 'report'})]
        try:
            salary = soup.find('dd', {'class': "col-sm-9 col-form-label pt-xso-0"}).text
        except:
            print('salary', code['jobname'])
            continue
        result = {'job': code['jobname'], 'salary': salary}
        values = [
            ' '.join([k.text for k in r.find_all('div', {'class' : "order-2 flex-grow-1"})])
                for r in soup.find_all('ul', {'class' : "list-unstyled m-0"})]
        for i in range(4):
            try:
                result[keys[i]] = values[i]
            except:
                print('etc', code['jobname'])
        
        with open('data/data19_en_onet.jsonl', 'a') as f:
            f.write(json.dumps(result) + '\n')

        
def crawl_mfm():
    for code in tqdm.tqdm(codes.to_dict('records')):
        link = f'https://www.mynextmove.org/profile/summary/{code["idx"]}'
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        try:
            text = soup.find('div', {'class' : "row mb-3 g-3"}).text
            definition = text.split('What they do:')[1].split('On the job, you would:')[0].strip("\n")
            explanation = text.split('On the job, you would:')[1].strip("\n")

        except Exception as e:
            print(e, code['jobname'])
            continue
        
        result = {'job': code['jobname'], 'link' : link, 'definition': definition, 'explanation': explanation}
        
        with open('data/data20_en_onet_mfm.jsonl', 'a') as f:
            f.write(json.dumps(result) + '\n')
        

if __name__ == '__main__':
    crawl_mfm()