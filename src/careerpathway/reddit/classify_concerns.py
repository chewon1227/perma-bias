import json
import sys
import os
from careerpathway.utils import open_json, save_json
import tqdm
sys.path.append('/home/chewon1227/careerpathway/src')
from careerpathway.llm import MyLLMEval

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def classify_concerns(data, model_name="Qwen/Qwen2.5-32B-Instruct"):
    output_file = "/home/chewon1227/careerpathway/data/reddit/classified_concerns.jsonl"
    
    system_prompt = "You are a classification expert. Your task is to classify concerns into EXACTLY one of these 6 categories: in-career, career-shift, after-grad, life-attitude, general-life, none."

    prompt = """Please classify the following concern:

{text}

Classification:
The categories are defined as follows:
1. in-career: Concerns about work life or university life, not general life concerns
2. career-shift: Concerns about career transition or changing careers
3. after-grad: Concerns about career path after graduation
4. life-attitude: Concerns about life attitude or how to live life
5. general-life: Concerns about general life concerns, not work or career concerns
6. none: Concerns that don't fit into any of the above categories

DO NOT ADD ANY EXPLANATION OR ADDITIONAL TEXT. JUST THE CATEGORY NAME."""
    
    # LLM 모델을 한 번만 초기화
    print("모델 로딩 중...")
    llm = MyLLMEval(model_name)
    print("모델 로딩 완료!")
        
    for i, item in enumerate(tqdm.tqdm(data, desc="분류 중")):
        
        post_text = item.get("Post Text", "")
        if not post_text:
            continue
            
        try:
            formatted_prompt = prompt.format(text=post_text)            
            input_data = [{
                'inputs': [formatted_prompt]
            }]
            
            result = llm.inference(
                prompt=formatted_prompt,
                data=input_data,
                system_prompt=system_prompt,
                max_tokens=50,
                temperature=0.0,
                save_path=None,
                apply_chat_template=False,
                save_additionally=False
            )
            
            # 결과가 None인지 확인
            if result is None or len(result) == 0:
                print(f"결과가 None입니다 (항목 {i+1})")
                item['classification'] = 'none'
            else:
                classification = result[0].strip().lower()            
                item['classification'] = classification
                # 처음 몇 개는 분류 결과 출력
                if i < 5:
                    print(f"항목 {i+1} 분류 결과: {classification}")
            
            # 바로바로 저장
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
        except Exception as e:
            print(f"오류 발생 (항목 {i+1}): {e}")
            item['classification'] = 'none'
            # 에러가 발생해도 저장
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"분류 완료! 결과가 {output_file}에 저장되었습니다.")

def sort_by_classification(data):
    classification_order = ['in-career', 'career-shift', 'after-grad', 'life-attitude', 'general-life', 'none']
    
    grouped = {}
    for item in data:
        classification = item.get('classification', 'none')
        if classification not in grouped:
            grouped[classification] = []
        grouped[classification].append(item)
    
    sorted_data = []
    for classification in classification_order:
        if classification in grouped:
            sorted_data.extend(grouped[classification])
    
    return sorted_data

def save_classified_data(data, output_path):
    save_json(data, output_path)

def main():
    input_file = "/home/chewon1227/careerpathway/data/reddit/all_with_comments.jsonl"
    
    print("JSONL 파일 로딩 중...")
    data = load_jsonl(input_file)
    print(f"총 {len(data)}개의 항목을 로드했습니다.")
    
    print("고민 분류 시작...")
    classify_concerns(data)
    
    print("분류 완료!")

if __name__ == "__main__":
    main()
