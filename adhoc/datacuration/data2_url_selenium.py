from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import fire
import json
from tqdm import tqdm
import time

QUERY_FORMAT = {
    'ko' : ["{} 되는 법", "{} 브이로그"],
    'en' : ["How to become {}", "{} vlog"],
    'ja' : ["{}になる方法", "{} vlog"],
    'sp' : ["Cómo convertirse en {}", "vlog de {}"]
}

# Function to perform YouTube search using Selenium and return video URLs and titles
def youtube_search(query, max_results=20):
    # Set up Selenium (using Chrome in this case)
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode (no UI)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    search_url = "https://www.youtube.com/results?search_query=" + query
    driver.get(search_url)

    # Wait for the search results to load
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "video-title")))
    except:
        print(f"Could not load search results for {query}")
        driver.quit()
        return []

    # Scroll down to load more results if needed
    for _ in range(max_results // 10):  # Assuming each scroll fetches roughly 10 results
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)  # Adjust sleep time if necessary

    # Scrape video titles and URLs
    video_elements = driver.find_elements(By.ID, "video-title")
    video_data = [(element.get_attribute("title"), element.get_attribute("href")) for element in video_elements if element.get_attribute("href")]

    driver.quit()

    return video_data[:max_results]  # Return only the top results (title and URL)


def make_urls(
    max_results : int = 20,
    query_format_idx : int = 0,
    language : str = 'ko',
    start: int = 0
    ):
    
    data = json.load(open('data/data5_jobdict.json', 'r'))
    query_format = QUERY_FORMAT[language][query_format_idx]
    i = 0
    for da in tqdm(data[start:]):
        job = da['jobname'][language]
        urls = youtube_search(query_format.format(job), max_results)
        
        # Save as CSV
        with open(f'data/data2_youtube_urls_{query_format_idx}.csv', 'a') as f:
            for title, url in urls:
                f.write(f"{start+i}\t{url}\t{title}\t{job}\t{language}\t{query_format}\n")
        i += 1


if __name__ == '__main__':
    fire.Fire(make_urls)
