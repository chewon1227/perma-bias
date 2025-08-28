from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get("https://www.work.go.kr/consltJobCarpa/srch/jobInfoSrch/srchJobInfo.do#scndCate")

job_links = []

for i in range(10):
    first_category = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, f"korSysJobA{i}"))
    )
    first_category.click()
    print(f"First Category: {i}")

    second_category_ul = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "categoryTwo"))
            )
    second_categories = second_category_ul.find_elements(By.TAG_NAME, "a")

    for second_category in second_categories:
        second_category.click()
        print(f"Second Category: {second_category.text}")
        time.sleep(1)

        result_links = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "srchResult"))
            )
        
        time.sleep(1)

        links = result_links.find_elements(By.TAG_NAME, "a")
        for link in links:
            print(f"링크 텍스트: {link.text}, 링크: {link.get_attribute('href')}")
            job_links.append({'text': link.text, 'href': link.get_attribute('href')})
        
        time.sleep(1)

driver.quit()

# Save the job links
import pickle
with open('data/12_jobdata/job_links.pkl', 'wb') as f:
    pickle.dump(job_links, f)

# crawl job data
import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

with open('data/12_jobdata/job_links.pkl', 'rb') as f:
    job_links = pickle.load(f)

job_data_list = []
for job_link in tqdm(job_links):
    url = job_link['href']

    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    job_data = {}

    job_data['직업명'] = job_link['text']

    job_description = soup.find('th', string='하는 일')
    if job_description:
        job_data['하는 일'] = job_description.find_next_sibling('td').text.strip()

    rows = soup.select('table tbody tr')

    for i in range(1, len(rows), 2):
        th_row = rows[i]
        td_row = rows[i + 1]

        th = th_row.find('th')

        if th is None:
            continue

        tds = td_row.find_all('td')

        if '교육/자격/훈련' in th.text and len(tds) > 0:
            job_data['관련학과'] = tds[0].text.strip()
            if '정확히 일치되는 학과정보가 없습니다.' in job_data['관련학과']:
                job_data['관련학과'] = None
            job_data['관련자격'] = tds[1].text.strip() if len(tds) > 1 else None
            if '자료가 존재하지 않습니다.' in job_data['관련자격']:
                job_data['관련자격'] = None
            job_data['훈련정보'] = tds[2].text.strip() if len(tds) > 2 else None

        elif '임금/직업만족도/전망' in th.text and len(tds) > 0:
            job_data['임금'] = tds[0].text.strip()
            job_data['직업만족도'] = tds[1].text.strip() if len(tds) > 1 else None
            job_data['전망'] = tds[2].text.strip() if len(tds) > 2 else None

        elif '능력/지식/환경' in th.text and len(tds) > 0:
            job_data['업무수행능력'] = tds[0].text.strip()
            job_data['지식'] = tds[1].text.strip() if len(tds) > 1 else None
            job_data['환경'] = tds[2].text.strip() if len(tds) > 2 else None

        elif '성격/흥미/가치관' in th.text and len(tds) > 0:
            job_data['성격'] = tds[0].text.strip()
            job_data['흥미'] = tds[1].text.strip() if len(tds) > 1 else None
            job_data['가치관'] = tds[2].text.strip() if len(tds) > 2 else None

        elif '업무활동' in th.text and len(tds) > 0:
            job_data['업무활동 중요도'] = tds[0].text.strip()
            job_data['업무활동 수준'] = tds[1].text.strip() if len(tds) > 1 else None

    # 일자리 현황
    job_status = soup.find('th', string='일자리 현황')
    if job_status:
        job_data['일자리 현황'] = job_status.find_next_sibling('td').text.strip()

    # 관련직업
    related_jobs = soup.find('th', string='관련직업')
    if related_jobs:
        job_data['관련직업'] = '\n'.join([job.text for job in related_jobs.find_next_sibling('td').find_all('a')])
        
    # 필요기술 및 지식
    necessary_skills = soup.find('p', string='필요기술 및 지식')
    if necessary_skills:
        necessary_skills_value = necessary_skills.find_next_sibling('div').text.strip()
        job_data['필요기술 및 지식'] = necessary_skills_value

    for key in job_data:
        # if the value is not a string, skip
        if not isinstance(job_data[key], str):
            continue
        # remove all tabs
        job_data[key] = job_data[key].replace('\t', '')
        # remove all 2+ newlines
        job_data[key] = re.sub(r'\n+', '\n', job_data[key])
    
    job_data_list.append(job_data)
    time.sleep(1)

# Save the job data
import json
with open('data/12_jobdata/job_data.json', 'w', encoding='utf-8') as f:
    json.dump(job_data_list, f, ensure_ascii=False, indent=2)
