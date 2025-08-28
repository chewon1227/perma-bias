import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from tqdm import trange


def careernet():
    base_url = "https://www.career.go.kr/cnet/front/base/job/jobInterview_new/jobSpecialList_new.do"
    base_urls = [base_url+f"?pageIndex={i}" for i in range(1, 40)]

    links = []
    for base_url in base_urls:
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all <a> tags related to page index
        interview_list = soup.find('div', {'class' : 'interview_list'})
        interview_links = interview_list.find_all('a', href=True)
        jobs = [job.text.split('\xa0\n')[0] for job in interview_list.find_all('strong', {'class' : 'interview_name'})]
        links.extend([{'link' : link['href'], 'jobname' : jobname} for link, jobname in list(zip(interview_links, jobs))])

    results = []

    for link in links:
        url = "https://www.career.go.kr" + link['link']
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract the job title
        job_title = soup.find('div', {'class': 'intvw_head'}).h3.text.split("\xa0")[1:]
        print(job_title)
        
        # Extract the company name
        company = soup.find('h3').text
        
        # Extract the interview questions
        questions = []
        for question in soup.find_all('div', {'class': 'question'}):
            questions.append(question.find('div', {'class': 'bubble'}).text)
        
        answers = []
        for answer in soup.find_all('div', {'class': 'answer'}):
            # answer is in the class='bubble' tag
            answers.append(answer.find('div', {'class': 'bubble'}).text)        
        
        results.append({
            **link,
            'questions': questions,
            'answers': answers,
        })
        
        with open('data/data10_careerinterview.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            

def _worknet_crawl(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    job_title = soup.find('span', {'class':'font-blue2'}).text
    
    questions = []
    for question in soup.find_all('dd', {'class': 'view_txt_q'}):
        questions.append(question.text.strip(''))
    
    answers = []
    for answer in soup.find_all('dd', {'class': 'view_txt'}):
        answers.append(answer.text.strip(''))
    
    
    if questions == [] and answers == []:
        
    return {
        'job_title': job_title,
        'url' : url,
        'questions': questions,
        'answers': answers,
    }
    
    
def worknet():
    results = []
    for i in range(1, 200):
        url = f'https://www.work.go.kr/consltJobCarpa/srch/meetPerson/meetPersonDetail.do?board_no=9&write_no={i}&pageIndex=12&pageUnit=5&searchYn=Y&chk_group=ALL&searchVal='
        try:
            results.append(_worknet_crawl(url))
            print(f"Successfully crawled {url}")
        except Exception as e:
            # print(e)
            pass
        with open('data/data10_careerinterview_2.json', 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    


def gather_result():
    json_paths = [
        'data/data10_careerinterview.json',
        'data/data10_careerinterview_2.json',
    ]
    
    jobs = []
    result = []
    for json_path in json_paths:
        print(f"Loaded {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            if item['job_title'] in jobs:
                continue
            if item['questions'] == [] and item['answers'] == []: 
                continue
            jobs.append(item['job_title'])
            result.append(item)
            
    print(len(result))
    with open('data/data10_careerinterview_final.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        

worknet()
gather_result()