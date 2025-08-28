import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
from time import sleep

COUNSELS_PAGES = 213
WEEKBEST_PAGES = 47

def crawl_counsels(weekbest = False, index = 1):
    if weekbest: # 이주의 공감상담
        base_url = f'https://www.career.go.kr/cnet/front/counsel/bestCaseList.do?pageIndex={index}'
        AGE_TAG = 'span'
        AGE_CLASS = 'first'
        Q_A_TAG = 'div'
        CLASS_Q = 'board-help-wrap'
        CLASS_Q_TITLE_TAG = 'strong'
        CLASS_Q_TITLE = 'help-title'
        CLASS_Q_CONTENT_TAG = 'div'
        CLASS_Q_CONTENT = 'help-cont-wrap'
        CLASS_A = 'board-help-wrap'
        CLASS_A_TITLE_TAG = 'div'
        CLASS_A_TITLE = 'heflp-reply_deff_title'
    else:
        base_url = f'https://www.career.go.kr/cnet/front/counsel/counselList.do?pageIndex={index}'
        AGE_TAG = 'li'
        AGE_CLASS = 'first'
        Q_A_TAG = 'table'
        CLASS_Q = 'board_view'
        CLASS_Q_TITLE_TAG = 'p'
        CLASS_Q_TITLE = 'view_title'
        CLASS_Q_CONTENT_TAG = 'td'
        CLASS_Q_CONTENT = 'view_comment'
        CLASS_A = 'advice_reply'
        CLASS_A_TITLE_TAG = 'th'
        CLASS_A_TITLE = 'reply_deff_title'

    data = []

    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    tbody = soup.find('tbody')
    tr = tbody.find_all('tr')
    tr = [t for t in tr if 'week_best' not in t.get('class', [])]
    post_links = [t.find('a')['href'] for t in tr]

    for a_post in post_links:
        sleep(0.5) # To prevent from being blocked

        post_url = 'https://www.career.go.kr' + a_post

        # if response.status_code != 200, retry until success
        while True:
            try:
                response = requests.get(post_url)
                response.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                print(f"HTTP error occurred: {e}")
                sleep(10)
                continue
        soup = BeautifulSoup(response.text, 'html.parser')

        # find <div class="board_view_conts">
        div = soup.find('div', {'class': 'board_view_conts'})

        post_question = div.find(Q_A_TAG, {'class': CLASS_Q})
        post_answer = div.find(Q_A_TAG, {'class': CLASS_A})

        post_question_age = post_question.find(AGE_TAG, {'class': AGE_CLASS}).text.removeprefix("대상 : ").strip()

        post_question_title = post_question.find(CLASS_Q_TITLE_TAG, {'class': CLASS_Q_TITLE}).get_text('\n', strip=True)
        post_question_content = post_question.find(CLASS_Q_CONTENT_TAG, {'class': CLASS_Q_CONTENT}).get_text('\n', strip=True)
        if weekbest:
            post_answer_title = div.find(CLASS_A_TITLE_TAG, {'class': CLASS_A_TITLE}).get_text('\n', strip=True)
            post_answer_content = div.find_all('div', {'class': 'help-cont-wrap'})[1]
            post_answer_content = '\n'.join(p.get_text('\n', strip=True) for p in post_answer_content.find_all('p')).replace(u'\xa0', u' ').strip()
        else:
            if post_answer:
                post_answer_title = post_answer.find(CLASS_A_TITLE_TAG, {'class': CLASS_A_TITLE}).get_text('\n', strip=True)
                post_answer_content = '\n'.join(p.get_text('\n', strip=True) for p in post_answer.find_all('p')).replace(u'\xa0', u' ').strip()
            else:
                post_answer_title = None
                post_answer_content = None

        data.append({
            'age': post_question_age,
            'q_title': post_question_title,
            'q_content': post_question_content,
            'a_title': post_answer_title,
            'a_content': post_answer_content
        })

    return data

data = crawl_counsels(weekbest=False, index=1)

with open('data/1_careernet/counsels-1stpage.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

data = crawl_counsels(weekbest=True, index=1)

with open('data/1_careernet/counsels-weekbest-1stpage.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

data = []
for i in tqdm(range(1, COUNSELS_PAGES + 1)):
    data += crawl_counsels(weekbest=False, index=i)

with open('data/1_careernet/counsels.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

data = []
for i in tqdm(range(1, WEEKBEST_PAGES + 1)):
    data += crawl_counsels(weekbest=True, index=i)

with open('data/1_careernet/counsels-weekbest.json', 'w') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
