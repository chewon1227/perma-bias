# question_base.jsonl -> generated_balance_questions.jsonl

import json
from itertools import combinations, product
import os
from multiprocessing import Pool, cpu_count
from functools import partial

def load_perma_data(jsonl_path):
    data_types = {
        "word": {},
        "factor": {},
        "reddit-negative": {},
        "reddit-positive": {},
        "reddit-com": {}
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            data_type = data["type"]
            if data_type in data_types:
                if data_type == "reddit-com":
                    # reddit-com은 key와 perma를 조합해서 키 생성
                    key = data.get("key", "")
                    perma = data["perma"]
                    data_types[data_type][f"{key}_{perma}"] = data
                else:
                    data_types[data_type][data["perma"]] = data
    
    return tuple(data_types.values())

def load_llm_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_question_record(question_data, question_id, **extra_fields):
    record = {
        "question_id": question_id
    }
    
    # key 필드를 question_id 바로 다음에 추가
    if "key" in question_data:
        record["key"] = question_data["key"]
    
    record.update({
        "question": question_data["question"],
        "factor1": question_data["factor1"],
        "factor2": question_data["factor2"],
        "op1": question_data["op1"],
        "op2": question_data["op2"],
        "op1_pos": question_data["op1_pos"],
        "op1_neg": question_data["op1_neg"],
        "op2_pos": question_data["op2_pos"],
        "op2_neg": question_data["op2_neg"],
        "label": question_data["label"]
    })

    # 나머지 추가 필드들 (reddit 타입 전용)
    for field in ["extent1", "extent2", "reddit_type"]:
        if field in question_data:
            record[field] = question_data[field]
    
    for key, value in extra_fields.items():
        record[key] = value
    
    return record

def generate_balance_questions_generic(data, extent_key="extent", desc_key="descriptions", 
                                     perma_key=None, job_placeholder=""):
    questions = []
    
    if not data:
        return questions
    
    data_combinations = list(combinations(data.keys(), 2))
    
    for factor1, factor2 in data_combinations:
        factor1_data = data[factor1]
        factor2_data = data[factor2]
        
        # extent 값 추출
        if extent_key == "extent":
            extent1_pos, extent1_neg = factor1_data[extent_key][0], factor1_data[extent_key][1]
            extent2_pos, extent2_neg = factor2_data[extent_key][0], factor2_data[extent_key][1]
        else:
            # factor 타입의 경우
            extent1_pos, extent1_neg = factor1_data[extent_key][0], factor1_data[extent_key][1]
            extent2_pos, extent2_neg = factor2_data[extent_key][0], factor2_data[extent_key][1]
        
        # descriptions 처리
        if desc_key == "descriptions":
            descs1 = [factor1_data[desc_key]] if isinstance(factor1_data[desc_key], str) else factor1_data[desc_key]
            descs2 = [factor2_data[desc_key]] if isinstance(factor2_data[desc_key], str) else factor2_data[desc_key]
        else:
            descs1 = [factor1_data.get(desc_key, "")]
            descs2 = [factor2_data.get(desc_key, "")]
        
        for desc1, desc2 in product(descs1, descs2):
            # word 타입은 extent 값 + perma 값으로 조합
            if desc_key == "extent":
                desc1_text = f"{extent1_pos} {factor1_data['perma']}"
                desc2_text = f"{extent2_pos} {factor2_data['perma']}"
                desc1_neg_text = f"{extent1_neg} {factor1_data['perma']}"
                desc2_neg_text = f"{extent2_neg} {factor2_data['perma']}"
            else:
                # factor, reddit 타입은 descriptions에서 하나씩 가져와서 조합
                desc1_text = f"{extent1_pos} {desc1}"
                desc2_text = f"{extent2_pos} {desc2}"
                desc1_neg_text = f"{extent1_neg} {desc1}"
                desc2_neg_text = f"{extent2_neg} {desc2}"
            
            question = f"1. {desc1_text} and {desc2_neg_text} {job_placeholder} vs. 2. {desc1_neg_text} and {desc2_text} {job_placeholder}"
            
            op1 = f"{desc1_text} and {desc2_neg_text}".lower()
            op2 = f"{desc1_neg_text} and {desc2_text}".lower()
            
            question_data = {
                "question": question.lower(),
                "factor1": factor1,
                "factor2": factor2,
                "op1": op1,
                "op2": op2,
                "op1_pos": desc1_text,
                "op1_neg": desc2_neg_text,
                "op2_pos": desc2_text,
                "op2_neg": desc1_neg_text,
                "label": f"{factor1} vs. {factor2}"
            }
            
            # 추가 필드들
            if desc_key != "descriptions" or not perma_key:
                question_data["desc1"] = desc1
                question_data["desc2"] = desc2
            
            questions.append(question_data)
    
    return questions


def process_factor_pair(args):
    _, _, factor1_data, factor2_data, job_placeholder = args
    questions = []
    
    # extent 값 추출
    factor1_min = factor1_data["extent"][1]
    factor1_max = factor1_data["extent"][0]
    factor2_min = factor2_data["extent"][1]
    factor2_max = factor2_data["extent"][0]
    
    # 경계선 탐색 질문들 (각 factor의 extent 범위 내에서)
    # op1_pos는 factor1의 2부터 최대값까지
    for op1_pos_num in range(2, factor1_max + 1):
        desc1_op1_pos = factor1_data["descriptions"].replace("{num}", str(op1_pos_num))
        
        # op2_pos는 factor2의 2부터 최대값까지
        for op2_pos_num in range(2, factor2_max + 1):
            desc2_op2_pos = factor2_data["descriptions"].replace("{num}", str(op2_pos_num))
            
            # op1_neg와 op2_neg는 무조건 1일 고정
            desc1_op1_neg = factor2_data["descriptions"].replace("{num}", str(factor2_min))
            desc2_op2_neg = factor1_data["descriptions"].replace("{num}", str(factor1_min))
            
            question = f"1. {desc1_op1_pos} and {desc1_op1_neg} {job_placeholder} vs. 2. {desc2_op2_neg} and {desc2_op2_pos} {job_placeholder}"
            
            # 새로운 구조: op1, op2, op1_pos, op1_neg, op2_pos, op2_neg
            op1 = f"{desc1_op1_pos} and {desc1_op1_neg}".lower()
            op2 = f"{desc2_op2_neg} and {desc2_op2_pos}".lower()
            
            questions.append({
                "question": question.lower(),
                "factor1": factor1_data["perma"],
                "factor2": factor2_data["perma"],
                "op1": op1,
                "op2": op2,
                "op1_pos": desc1_op1_pos,
                "op1_neg": desc1_op1_neg,
                "op2_pos": desc2_op2_pos,
                "op2_neg": desc2_op2_neg,
                "label": f"{factor1_data['perma']} vs {factor2_data['perma']}",
                "extent1": factor1_data["extent"],
                "extent2": factor2_data["extent"],
                "reddit_type": f"{factor1_data['perma'].lower()}-{factor2_data['perma'].lower()}",
                "key": factor1_data.get("key", "")
            })
    
    return questions

def generate_reddit_com_balance_questions(reddit_base_data, job_placeholder=""):
    """reddit-com 타입 데이터를 생성하여 경계선을 찾는 질문들을 만드는 함수
    병렬 처리를 사용하여 성능 향상
    
    Args:
        reddit_base_data: reddit-base 데이터
        job_placeholder: 직업 관련 플레이스홀더
    """
    if not reddit_base_data:
        return []
    
    reddit_base_combinations = list(combinations(reddit_base_data.keys(), 2))
    
    # 병렬 처리를 위한 인자 준비
    args_list = []
    for factor1, factor2 in reddit_base_combinations:
        factor1_data = reddit_base_data[factor1]
        factor2_data = reddit_base_data[factor2]
        args_list.append((factor1, factor2, factor1_data, factor2_data, job_placeholder))
    
    # 병렬 처리로 질문 생성
    num_processes = min(cpu_count(), 8)  # 최대 8개 프로세스 사용
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_factor_pair, args_list)
    
    # 결과 합치기
    all_questions = []
    for result in results:
        all_questions.extend(result)
    
    return all_questions

def main(question_type=None, key_filter=None):
    data_path = "/scratch/keummin/chaewon/data/question/question_base.jsonl"
    output_path = "/scratch/keummin/chaewon/data/question/generated_balance_questions.jsonl"
    
    if question_type:
        print(f"Generating {question_type} balance questions")
        print("=" * 60)
    else:
        print("Generating all balance questions")
        print("=" * 60)
    
    # 데이터 로드
    perma_data, factor_data, reddit_negative_data, reddit_positive_data, reddit_com_data = load_perma_data(data_path)
    
    print(f" 데이터 로드 완료:")
    print(f"  - PERMA (word): {len(perma_data)}개")
    print(f"  - Factor: {len(factor_data)}개") 
    print(f"  - Reddit-negative: {len(reddit_negative_data)}개")
    print(f"  - Reddit-positive: {len(reddit_positive_data)}개")
    print(f"  - Reddit-com: {len(reddit_com_data)}개")

    

    
    if question_type == "reddit-com":
        # key_filter가 지정된 경우 해당 key만 필터링
        if key_filter:
            filtered_data = {k: v for k, v in reddit_com_data.items() if v.get("key") == key_filter}
            print(f"🔍 Key '{key_filter}'로 필터링된 데이터: {len(filtered_data)}개")
            reddit_com_questions = generate_reddit_com_balance_questions(filtered_data)
        else:
            reddit_com_questions = generate_reddit_com_balance_questions(reddit_com_data)
        
        print(f"\n🔨 Reddit-com 질문 생성 완료: {len(reddit_com_questions)}개")
        
        with open(output_path, 'a', encoding='utf-8') as f:
            for question in reddit_com_questions:
                record = create_question_record(question, "reddit-com")
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"✅ Reddit-com 질문 {len(reddit_com_questions)}개가 {output_path}에 추가되었습니다!")
        return
    
    # 모든 타입의 질문들 생성 (통합된 함수 사용)
    word_questions = generate_balance_questions_generic(perma_data, extent_key="extent", desc_key="extent")
    factor_questions = generate_balance_questions_generic(factor_data, extent_key="extent", desc_key="descriptions")
    reddit_negative_questions = generate_balance_questions_generic(reddit_negative_data, extent_key="extent", desc_key="descriptions")
    reddit_positive_questions = generate_balance_questions_generic(reddit_positive_data, extent_key="extent", desc_key="descriptions")
    reddit_com_questions = generate_reddit_com_balance_questions(reddit_com_data)
    
    print(f"질문 생성 완료:")
    print(f"  - Word: {len(word_questions)}개")
    print(f"  - Factor: {len(factor_questions)}개")
    print(f"  - Reddit-negative: {len(reddit_negative_questions)}개")
    print(f"  - Reddit-positive: {len(reddit_positive_questions)}개")
    print(f"  - Reddit-com: {len(reddit_com_questions)}개")
    
    # 모든 질문들을 하나의 파일에 저장
    all_questions = []
    
    question_groups = [
        (word_questions, "word"),
        (factor_questions, "factor"),
        (reddit_negative_questions, "reddit-negative", {"polarity": "negative"}),
        (reddit_positive_questions, "reddit-positive", {"polarity": "positive"}),
        (reddit_com_questions, "reddit-com")
    ]
    
    for group in question_groups:
        questions, question_id = group[0], group[1]
        extra_fields = group[2] if len(group) > 2 else {}
        
        for question in questions:
            record = create_question_record(question, question_id, **extra_fields)
            all_questions.append(record)
    
    # 파일에 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in all_questions:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"파일 저장 완료: {output_path}")
    print(f"총 {len(all_questions)}개의 balance questions가 생성되었습니다!")
    print("모든 질문 생성 완료!")


if __name__ == "__main__":
    import sys
    
    # 명령행 인자로 특정 타입과 key_filter 지정 가능
    if len(sys.argv) > 1:
        question_type = sys.argv[1]
        key_filter = sys.argv[2] if len(sys.argv) > 2 else None
        
        if question_type in ['word', 'factor', 'reddit-negative', 'reddit-positive', 'reddit-base', 'reddit-com']:
            main(question_type, key_filter)
        
    else:
        # 인자가 없으면 모든 타입 처리
        main()
