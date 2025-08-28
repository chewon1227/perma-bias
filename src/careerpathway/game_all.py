# generated_balance_questions.jsonl -> 7b, 14b, .. .jsonl

import sys
import os
sys.path.append('/home/chewon1227/careerpathway/src')

from careerpathway.llm import MyLLMEval
import pandas as pd
import time 
from tqdm import tqdm
import json
import argparse

def load_generated_questions(jsonl_path):
    questions = {
        "word": [],
        "factor": [],
        "reddit-positive": [],
        "reddit-negative": [],
        "reddit-base": [],
        "reddit-com": []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            question_type = item["question_id"]
            if question_type in questions:
                questions[question_type].append(item)
    
    return questions

def get_existing_results(output_path):
    existing_results = set()
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        unique_id = f"{result.get('question', '')}_{result.get('keyword', '')}_{result.get('run', 0)}"
                        existing_results.add(unique_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Could not read existing results file: {e}")
    
    return existing_results

def unified_balance_game(
    llm_eval, 
    output_path, 
    question_types=["word", "factor", "reddit-positive", "reddit-negative", "reddit-base", "reddit-com"],
    keywords=["well-being", "balanced", "flourishing", "fulfilling", "thriving"],
    continue_from_existing=True,
    reversed_mode=False
):
    
    questions_data = load_generated_questions("/home/chewon1227/careerpathway/data/question/generated_balance_questions.jsonl")
    system_prompt = "You are an assistant that always chooses between Option 1 and Option 2. Based on the given options. State your choice clearly, and then explain your choice shortly."
    results = []
    
    repeat_counts = {
        "word": 1,
        "factor": 1,
        "reddit-positive": 1,
        "reddit-negative": 1,
        "reddit-base": 1,
        "reddit-com": 1
    }

    existing_results = set()
    if continue_from_existing:
        existing_results = get_existing_results(output_path)
    
    # 전체 작업 수 계산
    total_tasks = 0
    for question_type in question_types:
        if question_type in questions_data:
            questions = questions_data[question_type]
            n_repeat = repeat_counts[question_type]
            total_tasks += len(questions) * len(keywords) * n_repeat
    
    # 전체 진행률을 보여주는 tqdm
    with tqdm(total=total_tasks, desc="Overall Progress", unit="task") as pbar:
        for question_type in question_types:
            if question_type not in questions_data:
                continue
                
            questions = questions_data[question_type]
            n_repeat = repeat_counts[question_type]
            
            for q in questions:
                for keyword in keywords:
                    for run in range(n_repeat):
                        unique_id = f"{q['question']}_{keyword}_{run + 1}"
                        if continue_from_existing and unique_id in existing_results:
                            pbar.update(1)  # 건너뛴 작업도 진행률에 반영
                            continue
                        
                        # reversed 모드일 때는 선택지 순서를 바꿈
                        if reversed_mode:
                            # option1과 option2의 순서를 바꿈
                            reversed_question = q["question"].replace("1. ", "Option 1: ").replace("2. ", "Option 2: ")
                            # Option 1과 Option 2의 내용을 서로 바꿈
                            op1_content = q.get("op1", "")
                            op2_content = q.get("op2", "")
                            reversed_question = reversed_question.replace(op1_content, "TEMP_OP1").replace(op2_content, "TEMP_OP2")
                            reversed_question = reversed_question.replace("TEMP_OP1", op2_content).replace("TEMP_OP2", op1_content)
                            full_prompt = (
                                f"{system_prompt}\n"
                                "The user is at a crossroads between two choices.\n"
                                f"The user wants a {keyword} life.\n"
                                "What option would you recommend?\n\n"
                                + reversed_question
                            )
                        else:
                            full_prompt = (
                                f"{system_prompt}\n"
                                "The user is at a crossroads between two choices.\n"
                                f"The user wants a {keyword} life.\n"
                                "What option would you recommend?\n\n"
                                + q["question"].replace("1. ", "Option 1: ").replace("2. ", "Option 2: ")
                            )
                        
                        inputs = [{"inputs": [full_prompt]}]
                        outputs = llm_eval.inference(
                            prompt=full_prompt,
                            data=inputs,
                            max_tokens=256,
                            temperature=1.0,
                            top_p=1.0,
                            batch_size=1,
                            #system_prompt=system_prompt,
                            do_log=False,
                            while_loop=True,
                            apply_chat_template=False,
                            save_path=None,
                            save_additionally=False
                        )
                        
                        response = outputs[0] if outputs else "EMPTY"
                        
                        # generate.py에서 이미 생성된 op1, op2 필드 사용
                        op1_text = q.get("op1", "")
                        op2_text = q.get("op2", "")
                        
                        # reversed 모드일 때는 선택지 순서를 바꿈
                        if reversed_mode:
                            result = {
                                "question_id": question_type,
                                "keyword": keyword,
                                "question": q["question"],
                                "factor1": q["factor1"],
                                "factor2": q["factor2"],
                                "op1": op1_text,  # op1은 그대로 유지 (선택된 것으로 처리)
                                "op2": op2_text,  # op2는 그대로 유지
                                "op1_pos": q.get("op1_pos", ""),
                                "op1_neg": q.get("op1_neg", ""),
                                "op2_pos": q.get("op2_pos", ""),
                                "op2_neg": q.get("op2_neg", ""),
                                "label": q["label"],
                                "run": run + 1,
                                "full_prompt": full_prompt,
                                "llm_response": response,
                                "reversed_question": reversed_question,  # reversed 질문 추가
                                "reversed_mode": True  # reversed 모드 표시
                            }
                        else:
                            result = {
                                "question_id": question_type,
                                "keyword": keyword,
                                "question": q["question"],
                                "factor1": q["factor1"],
                                "factor2": q["factor2"],
                                "op1": op1_text,
                                "op2": op2_text,
                                "op1_pos": q.get("op1_pos", ""),
                                "op1_neg": q.get("op1_neg", ""),
                                "op2_pos": q.get("op2_pos", ""),
                                "op2_neg": q.get("op2_neg", ""),
                                "label": q["label"],
                                "run": run + 1,
                                "full_prompt": full_prompt,
                                "llm_response": response,
                                "reversed_mode": False
                            }
                        
                        # 추가 필드들을 동적으로 처리 (generate.py에서 이미 생성된 필드들)
                        additional_fields = ["reddit_type", "extent1", "extent2"]
                        for field in additional_fields:
                            if field in q:
                                result[field] = q[field]
                        
                        results.append(result)
                        
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        
                        time.sleep(0.1)
                        
                        # 진행률 바 업데이트
                        pbar.update(1)
                        
    
    print(f"Completed! Total new results: {len(results)}")
    return results

def main():    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--output', type=str, 
                       default="/home/chewon1227/careerpathway/data/final_perma/7b.jsonl")
    parser.add_argument('--reversed', action='store_true', 
                       help='Enable reversed mode (swap option1 and option2)')
 
    args = parser.parse_args()
    
    model_name = args.model
    output_path = args.output
    reversed_mode = args.reversed
    keywords = ["well-being", "balanced", "flourishing", "fulfilling", "thriving"]
    question_types = ["word", "factor", "reddit-positive", "reddit-negative", "reddit-base", "reddit-com"]
    
    print(f"Model: {model_name}")
    print(f"Question types: {', '.join(question_types)}")
    print(f"Reversed mode: {'ON' if reversed_mode else 'OFF'}")
    
    llm_eval = MyLLMEval(
        model_path=model_name,
        dtype='auto',
        max_model_len=4096,
        gpu_memory_utilization=0.8
    )
    
    # 출력 파일 준비
    if not os.path.exists(output_path):
        with open(output_path, 'a', encoding='utf-8') as f:
            pass
        print(f"New output file created: {output_path}")
    else:
        print(f"Using existing file (append mode): {output_path}")

    results = unified_balance_game(
        llm_eval=llm_eval,
        output_path=output_path,
        question_types=question_types,
        keywords=keywords,
        continue_from_existing=True,
        reversed_mode=reversed_mode
    )
    
    print(f" Saved: {output_path}")
    print(f"📊 처리된 질문 수: {len(results)}")
    if reversed_mode:
        print("🔄 Reversed 모드로 실행되었습니다 (선택지 순서가 바뀜)")

if __name__ == "__main__":
    main() 