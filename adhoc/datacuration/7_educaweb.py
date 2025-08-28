import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from tqdm import tqdm
import fire

def _get_educawev(page_content):
    # Step 2: Parse the page with BeautifulSoup
    soup = BeautifulSoup(page_content, 'html.parser')

    try:
        descripcion_section = soup.find('section', id='section-characteristics')
        descripcion_content = descripcion_section.find('div', {'class':'por'}).text.split("ver más")[0]
    except:
        descripcion_content = None
        
    try:
        temario_section = soup.find('div', id='section-program')
        temario_content = temario_section.find('div', {'class' : 'por'}).text.split("ver más")[0]
    except:
        temario_content = None
        
    try:
        requisitos_section = soup.find('div', id='section-requirements')
        requisitos_content = requisitos_section.find('div', {'class' : 'por'}).text.split("ver más")[0]
    except:
        requisitos_content = None
        
    return descripcion_content, temario_content, requisitos_content


def educaweb_dict(start: int = 0):
    base_url = "https://www.educaweb.com/profesiones/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    subcategories = soup.find_all('a', {'class' : "card-v1__title"})
    links = [sub.get('href') for sub in subcategories]
    big_category = [sub.text for sub in subcategories]
    
    result = []
    i = 0
    for link, big in zip(links, big_category):
        url = f'https://www.educaweb.com/{link}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        counts = int(soup.find('span', {'class' : "o-banner__h1--cont"}).text)
        sub_links = []
        for page_id in range(1, counts//20+2):
            url = f'https://www.educaweb.com/{link}/?p={page_id}'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            sub_links += [sub.get('href') for sub in soup.find_all('a', {'class' : "sz-result__t"})]
        print(f"big category: {big}, length: {len(sub_links)}, counts: {counts}")
                                                        
        for sub_link in tqdm(sub_links):
            url = f'https://www.educaweb.com/{sub_link}'
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            job = soup.find('h1', class_='banner__t').text.strip()
            keys = soup.find_all('h2', "o-box__title o-box__title--t c0")
            keys = [k.text.split(" de")[0].split(" ser")[0].split(f" un {job.lower()}")[0].split(f" una {job.lower()}")[0] for k in keys]
            descriptions = soup.find_all('div', class_='o-list--pre')
            descriptions = [r.text for r in descriptions]
            print(len(descriptions))
            
            single_result = {
                'big_category': big,
                'job': job,
                'url': url,
                **{k.strip(): v.strip() for k, v in zip(keys, descriptions)}
            }
            with open('data/data7_educaweb.json', 'a') as f:
                json.dump(single_result, f, indent=2, ensure_ascii=False)
                f.write(',\n')  # Optionally add a newline for better readability
            print('done', job)                      
                
                
                        
def educaweb_course(start: int = 0):
    base_url = "https://www.educaweb.com/profesiones/"
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    subcategories = soup.find_all('a', {'class' : "card-v1__title"})
    links = [sub.get('href') for sub in subcategories]
    
    result = []
    i = 0
    for link in links:
        url = f'https://www.educaweb.com/{link}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        sub_links = [sub.get('href') for sub in soup.find_all('a', {'class' : 'o-btn'})]
        
        for sub_link in tqdm(sub_links):
            url = f'https://www.educaweb.com/{sub_link}'
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            subsubs = soup.find_all('a', {'class' : 'sz-result__t'})
            subsub_links = [subsub.get('href') for subsub in subsubs]
            subsub_titles = [subsub.get('title') for subsub in subsubs]
            
            for subsub_link, subsub_title in zip(subsub_links, subsub_titles):
                if i < start:
                    i += 1
                    # print('already done,', subsub_title)
                    continue
                else:
                    i += 1
                    url = f'https://www.educaweb.com/{subsub_link}'
                    response = requests.get(url)
                    if response.status_code == 200:
                        page_content = response.text
                        descripcion_content, temario_content, requisitos_content = _get_educawev(page_content)
                        
                        single_result = {
                            'title': subsub_title,
                            'url': url,
                            'description': descripcion_content,
                            'tema': temario_content,
                            'requirements': requisitos_content
                        }

                        # Open the file in append mode and write each entry as a new JSON object
                        with open('data/data7_educaweb.json', 'a') as f:
                            json.dump(single_result, f, indent=2, ensure_ascii=False)
                            f.write(',\n')  # Optionally add a newline for better readability

                        print('done', subsub_title)                      
                    else:
                        print(f"Failed to retrieve the page. Status code: {response.status_code}")
                    
          
if __name__ == '__main__':          
    fire.Fire(educaweb_dict)