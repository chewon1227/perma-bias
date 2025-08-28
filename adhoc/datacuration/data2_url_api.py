# from googleapiclient.discovery import build
import os
# from mylmeval import open_json
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import fire
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from oauth2client.tools import argparser
import time

# Replace with your own API key
API_KEY = os.getenv('GOOGLE_API_KEY')
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
QUERY_FORMAT = {
    'ko' : ["{} 되는 법", "{} 브이로그"],
    'en' : ["How to become {}", "{} vlog"],
    'ja' : ["{}になる方法", "{} vlog"],
    'sp' : ["Cómo convertirse en {}", "vlog de {}"]
}



def youtube_search(query, max_results=20):
    # Build the YouTube service
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
    
    try:
        # Perform the search
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=max_results
        ).execute()
        
        # Parse the results to extract video titles and URLs
        video_data = []
        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                video_id = search_result['id']['videoId']
                video_title = search_result['snippet']['title']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Append both title and URL as a tuple
                video_data.append((video_title, video_url))
        
        return video_data
    
    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")

            
def make_urls(
    max_results : int = 20,
    language : str = 'ko',
    start: int = 0
    ):
    
    data = json.load(open('data/data5_jobdict.json', 'r'))
    query_format = QUERY_FORMAT[language]
    i = 0
    for da in tqdm(data[start:]):
        job = da['jobname'][language]
        urls = youtube_search(query_format.format(job), max_results)
        
        # Save as CSV
        with open(f'data/data2_youtube_urls.csv', 'a') as f:
            for title, url in urls:
                f.write(f"{start+i}\t{url}\t{title}\t{job}\t{language}\n")
        time.sleep(1)
        i += 1


if __name__ == '__main__':
    fire.Fire(make_urls)
