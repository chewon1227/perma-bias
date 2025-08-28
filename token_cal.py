import tiktoken
import json
import os

# gpt-o1-preview는 cl100k_base 인코딩을 사용할 것으로 추정
# (GPT-4 계열과 동일한 토크나이저)
encoder = tiktoken.get_encoding("cl100k_base")

file_path = "/home/chewon1227/careerpathway/data/data3_reddit_data.jsonl"
total_tokens = 0
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            text = json.dumps(data, ensure_ascii=False)
            tokens = len(encoder.encode(text))
            total_tokens += tokens
        except json.JSONDecodeError:
            print(f"JSON 파싱 오류 발생: {line[:100]}...")

# GPT-o1-preview 가격 (정확한 가격은 확인 필요)
# 임시로 GPT-4o-mini 가격 사용
cost_per_1k_tokens = 15/1000000  # $0.00015 per 1K tokens (실제 o1 가격은 다를 수 있음)
total_cost = (total_tokens / 1000) * cost_per_1k_tokens

print(f"총 토큰 수: {total_tokens:,}")
print(f"예상 비용: ${total_cost:.4f}")
print("참고: gpt-o1-preview의 실제 가격은 다를 수 있습니다.")
