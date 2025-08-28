import json
from tqdm import tqdm

input_path = '/home/chewon1227/careerpathway/data/reddit/all_with_comments.jsonl'  # 기존 파일 경로
output_path = '/home/chewon1227/careerpathway/data/reddit/300over_comments.jsonl' # 저장할 파일 경로

def filter_posts_by_comment_count(input_path, output_path, min_comments=5):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing"):
            try:
                post = json.loads(line)
                if "comments" in post and len(post["comments"]) >= min_comments:
                    json.dump(post, outfile)
                    outfile.write('\n')
            except json.JSONDecodeError:
                continue  # malformed JSON line, skip

# 실행
filter_posts_by_comment_count(input_path=input_path, output_path=output_path, min_comments=300)