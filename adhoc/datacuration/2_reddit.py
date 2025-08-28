import os
import praw
import datetime
import fire
import pandas as pd
from tqdm import tqdm
import json
import time
import queue
import itertools
import concurrent.futures
import locale
from praw.exceptions import PRAWException
from prawcore import ResponseException, RequestException

def clean_text(text):
    """텍스트 데이터 정제 및 인코딩 처리"""
    if text is None:
        return ""
    try:
        # 인코딩 이슈 있는 문자 필터링
        text = ''.join(char for char in str(text) if ord(char) < 65536)
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        print(f"Error cleaning text: {e}")
        return ""

def create_reddit_instance():
    """안전한 Reddit 인스턴스 생성"""
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        
        return praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT'),
            check_for_updates=False,
            check_for_async=False,
            requestor_kwargs={'timeout': 30}
        )
    except Exception as e:
        print(f"Error creating Reddit instance: {e}")
        raise

def handle_reddit_error(e, context=""):
    """Reddit API 에러 처리"""
    if isinstance(e, RequestException):
        if "latin-1" in str(e):
            print(f"Encoding error in {context}: Please check your environment variables and encoding settings")
        else:
            print(f"Reddit API request error in {context}: {e}")
    elif isinstance(e, ResponseException):
        print(f"Reddit API response error in {context}: {e}")
    else:
        print(f"Unexpected error in {context}: {e}")
    return None

def append_to_jsonl(data, filename):
    """단일 데이터를 JSONL 파일에 추가"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'a', encoding='utf-8') as f:
            cleaned_data = {
                k: clean_text(v) if isinstance(v, str) else v
                for k, v in data.items()
            }
            
            if 'Comments' in cleaned_data:
                cleaned_data['Comments'] = [
                    {
                        'score': c['score'],
                        'comment': clean_text(c['comment'])
                    }
                    for c in cleaned_data['Comments']
                ]
            
            json_str = json.dumps(cleaned_data, ensure_ascii=False)
            f.write(json_str + '\n')
    except Exception as e:
        print(f"Error appending to file {filename}: {e}")

def fetch_comments_with_rate_limit(reddit, url, comment_n=5, max_retries=3):
    """댓글 가져오기 with 재시도 로직"""
    for attempt in range(max_retries):
        try:
            submission = reddit.submission(url=url)
            submission.comment_sort = 'top'
            submission.comments.replace_more(limit=0)
            comments = submission.comments.list()
            comments = [
                {
                    'score': r.score,
                    'comment': clean_text(str(r.body))
                }
                for r in comments 
                if '[deleted]' not in r.body
            ][:comment_n]
            time.sleep(1)
            return comments
        except (ResponseException, RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = 60
                print(f"Rate limit hit. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Max retries reached for URL: {url}")
                return []
        except Exception as e:
            print(f"Error fetching comments for {url}: {e}")
            return []

def scrap_reddit_batch(
    keyword,
    sort_type='new',
    start=0,
    n=10000,
    do_comment=False,
    comment_n=5,
    batch_size=25
):
    """배치 모드 스크래핑"""
    try:
        for env_var in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']:
            if not os.getenv(env_var):
                raise ValueError(f"Missing environment variable: {env_var}")

        reddit = create_reddit_instance()
        subreddit = reddit.subreddit(keyword)
        posts_processed = 0
        output_path = f'data/data3_reddit_data_{keyword}.jsonl'
        
        if sort_type == 'hot':
            post_iterator = subreddit.hot(limit=None)
        else:
            post_iterator = subreddit.new(limit=None)
        
        if start > 0:
            print(f"Skipping to position {start}...")
            for _ in tqdm(range(start)):
                try:
                    next(post_iterator)
                except StopIteration:
                    print("Reached end of posts before start position")
                    return
        
        with tqdm(total=n) as pbar:
            while posts_processed < n:
                try:
                    posts = list(itertools.islice(post_iterator, batch_size))
                    if not posts:
                        break
                    
                    for post in posts:
                        post_data = {
                            "Title": clean_text(post.title),
                            "Post Text": clean_text(post.selftext),
                            "ID": post.id,
                            "Score": post.score,
                            "Total Comments": post.num_comments,
                            "Post URL": post.url,
                            "Created UTC": post.created_utc,
                            "Position": start + posts_processed
                        }
                        
                        if do_comment:
                            comments = fetch_comments_with_rate_limit(reddit, post.url, comment_n=comment_n)
                            post_data['Comments'] = comments
                        
                        append_to_jsonl(post_data, output_path)
                        posts_processed += 1
                        pbar.update(1)
                        
                        if posts_processed >= n:
                            break
                    
                    time.sleep(1)
                    
                except (ResponseException, RequestException) as e:
                    wait_time = 60
                    print(f"Rate limit hit. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                except StopIteration:
                    print("Reached end of available posts")
                    break
                except Exception as e:
                    print(f"Error occurred at position {start + posts_processed}: {str(e)}")
                    continue
        
        print(f"Successfully processed {posts_processed} posts starting from position {start}")
        print(f"Data saved to {output_path}")
        
    except Exception as e:
        print(f"Critical error in batch processing: {e}")

def scrap_reddit_parallel(
    keyword,
    sort_type='new',
    start=0,
    n=10000,
    do_comment=False,
    comment_n=5,
    max_workers=3
):
    """병렬 처리 모드 스크래핑"""
    for env_var in ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']:
        if not os.getenv(env_var):
            raise ValueError(f"Missing environment variable: {env_var}")

    reddit = create_reddit_instance()
    post_queue = queue.Queue()
    output_path = f'data/data3_reddit_data_{keyword}.jsonl'
    processed_count = 0

    def worker():
        nonlocal processed_count
        while processed_count < n:
            try:
                post = post_queue.get_nowait()
            except queue.Empty:
                break

            try:
                title = post.title if hasattr(post, 'title') else ''
                selftext = post.selftext if hasattr(post, 'selftext') else ''
                
                post_data = {
                    "Title": clean_text(title),
                    "Post Text": clean_text(selftext),
                    "ID": str(post.id) if hasattr(post, 'id') else '',
                    "Score": int(post.score) if hasattr(post, 'score') else 0,
                    "Total Comments": int(post.num_comments) if hasattr(post, 'num_comments') else 0,
                    "Post URL": str(post.url) if hasattr(post, 'url') else '',
                    "Created UTC": float(post.created_utc) if hasattr(post, 'created_utc') else 0,
                    "Position": start + processed_count
                }
                
                if do_comment:
                    comments = fetch_comments_with_rate_limit(reddit, post_data["Post URL"], comment_n=comment_n)
                    post_data['Comments'] = comments

                append_to_jsonl(post_data, output_path)
                processed_count += 1
            except Exception as e:
                handle_reddit_error(e, f"processing post {getattr(post, 'id', 'unknown')}")
            finally:
                post_queue.task_done()
                time.sleep(1)

    if sort_type == 'hot':
        post_iterator = reddit.subreddit(keyword).hot(limit=n)
    else:
        post_iterator = reddit.subreddit(keyword).new(limit=n)

    if start > 0:
        print(f"Skipping to position {start}...")
        for _ in tqdm(range(start)):
            try:
                next(post_iterator)
            except StopIteration:
                print("Reached end of posts before start position")
                return

    posts = list(itertools.islice(post_iterator, n))
    for post in posts:
        post_queue.put(post)

    print(f"Starting parallel processing with {max_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker) for _ in range(max_workers)]
        
        with tqdm(total=n) as pbar:
            while processed_count < n:
                last_count = processed_count
                time.sleep(0.1)
                pbar.update(processed_count - last_count)
                if all(future.done() for future in futures):
                    break

    print(f"Successfully processed {processed_count} posts starting from position {start}")
    print(f"Data saved to {output_path}")
        

if __name__ == "__main__":
    fire.Fire({
        'batch': scrap_reddit_batch,
        'parallel': scrap_reddit_parallel
    })