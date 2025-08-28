import json
from tqdm import tqdm
from careerpathway.llm import MyLLMEval

# 모델 및 프롬프트 설정
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
llm = MyLLMEval(model_path=MODEL_NAME)

PROMPT = (
    "You're given a Reddit post and a comment in response. Classify which PERMA domain the comment primarily reflects: "
    "Positive Emotion, Engagement, Relationships, Meaning, or Accomplishment. "
    "If not related, return 'None'.\n\n"
    "Post:\n{0}\n\n"
    "Comment:\n{1}\n\n"
    "Which PERMA domain does the comment reflect most? Return one of: "
    "Positive Emotion, Engagement, Relationships, Meaning, Accomplishment, None."
)

# 입력/출력 파일 경로
INPUT_FILE = "/home/chewon1227/careerpathway/data/reddit/5over_comments.jsonl"
OUTPUT_FILE = "/home/chewon1227/careerpathway/data/reddit/results.jsonl"

with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
    for line in tqdm(infile, desc="Processing posts"):
        post = json.loads(line)
        post_text = post.get("Post Text", "")
        comments = post.get("comments", [])

        data_batch = []
        comment_id_map = {}

        for comment in comments:
            comment_text = comment.get("body", "")[:300]
            if not comment_text.strip():
                continue

            comment_id = comment.get("id", "")
            data_batch.append({
                "inputs": [post_text[:200], comment_text],
                "meta": {"comment_id": comment_id}
            })
            comment_id_map[comment_id] = comment

        # 댓글이 없다면 그대로 저장
        if not data_batch:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
                json.dump(post, outfile)
                outfile.write('\n')
            continue

        try:
            results = llm.inference(
                prompt=PROMPT,
                data=data_batch,
                max_tokens=5,
                temperature=0.0,
                top_p=1.0,
                apply_chat_template=False,
                system_prompt=None,
                batch_size=1,
                save_additionally=False,
                while_loop=True
            )

            for item, result in zip(data_batch, results):
                cid = item["meta"]["comment_id"]
                if cid in comment_id_map:
                    comment_id_map[cid]["perma"] = result

        except Exception as e:
            for item in data_batch:
                cid = item["meta"]["comment_id"]
                if cid in comment_id_map:
                    comment_id_map[cid]["perma"] = f"ERROR: {str(e)}"

        # ➤ post 단위로 바로 저장 (with open 한 줄 안에!)
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
            json.dump(post, outfile)
            outfile.write('\n')