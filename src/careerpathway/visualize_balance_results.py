import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from pathlib import Path

def extract_numbers_from_text(text):
    """텍스트에서 첫 번째 숫자를 추출하는 함수"""
    if not text:
        return None
    
    numbers = re.findall(r'\d+', text)
    return int(numbers[0]) if numbers else None

def extract_choice(text):
    """LLM 응답에서 선택을 추출하는 함수 (1 또는 2 반환)"""
    text = str(text)
    idx1 = text.find("Option 1")
    idx2 = text.find("Option 2")

    if idx1 == -1 and idx2 == -1:
        return "unknown"
    elif idx1 != -1 and idx2 != -1:
        return "1" if idx1 < idx2 else "2"
    elif idx1 != -1:
        return "1"
    else:
        return "2"

def analyze_balance_results(file_path):
    """밸런스 게임 결과 데이터를 분석하는 함수"""
    if not Path(file_path).exists():
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return []
    
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 라인 {line_num} JSON 파싱 오류: {e}")
                    continue
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return []
    
    print(f"📊 데이터 로드 완료: {len(data)}개 항목")
    
    # 데이터 분석
    analysis_results = []
    
    for item in data:
        op1_pos_num = extract_numbers_from_text(item.get('op1_pos', ''))
        op1_neg_num = extract_numbers_from_text(item.get('op1_neg', ''))
        op2_pos_num = extract_numbers_from_text(item.get('op2_pos', ''))
        op2_neg_num = extract_numbers_from_text(item.get('op2_neg', ''))
        
        llm_response = item.get('llm_response', '')
        choice = extract_choice(llm_response)  # 직접 "1" 또는 "2" 사용
        
        if all(x is not None for x in [op1_pos_num, op1_neg_num, op2_pos_num, op2_neg_num]):
            analysis_results.append({
                'op1_pos_num': op1_pos_num,
                'op1_neg_num': op1_neg_num,
                'op2_pos_num': op2_pos_num,
                'op2_neg_num': op2_neg_num,
                'choice': choice,  # "1" 또는 "2"
                'keyword': item.get('keyword', ''),
                'question_id': item.get('question_id', ''),
                'reddit_type': item.get('reddit_type', ''),
                'question': item.get('question', '')[:100] + '...' if len(item.get('question', '')) > 100 else item.get('question', '')
            })
    
    print(f"✅ 분석 완료: {len(analysis_results)}개 유효한 결과")
    return analysis_results

def create_visualization(data, output_path=None):
    """데이터를 시각화하는 메인 함수"""
    if not data:
        print("❌ 시각화할 데이터가 없습니다.")
        return None
    
    df = pd.DataFrame(data)
    
    # keyword별로 데이터 분류
    keywords = df['keyword'].unique()
    print(f"🔍 키워드별 분석 시작: {keywords}")
    
    # 각 keyword별로 분석
    for keyword in keywords:
        keyword_data = df[df['keyword'] == keyword].copy()
        if len(keyword_data) == 0:
            continue
        
        keyword_output_path = f"{keyword}" if output_path else None
        print(f"📊 {keyword} 분석 중... ({len(keyword_data)}개 데이터)")
        
        # 통합 시각화 생성
        create_unified_visualization(keyword_data, keyword, keyword_output_path)
    
    return df

def create_unified_visualization(df, keyword, output_path):
    """키워드별 통합 시각화 생성"""
    # 히트맵으로 선택 패턴 시각화
    plt.figure(figsize=(12, 10))
    
    # op1_pos와 op2_pos 값들을 정렬
    op1_values = sorted(df['op1_pos_num'].unique())
    op2_values = sorted(df['op2_pos_num'].unique())
    
    # 선택을 1과 2로 변환 (이미 "1", "2"이므로 직접 사용)
    choice_numeric = df['choice'].map({'1': 1, '2': 2, 'unknown': np.nan})
    
    # 유효한 선택만 필터링
    valid_choices = df[df['choice'].isin(['1', '2'])].copy()
    
    if len(valid_choices) > 0:
        # 피벗 테이블 생성 - 각 op1_pos, op2_pos 조합에서의 선택
        pivot_data = valid_choices.pivot_table(
            values='choice', 
            index='op1_pos_num', 
            columns='op2_pos_num', 
            aggfunc=lambda x: choice_numeric[x.index].mode()[0] if len(x) > 0 else np.nan,
            fill_value=np.nan
        )
        
        # 히트맵 생성 - 1과 2로 표시
        sns.heatmap(pivot_data, annot=True, cmap=['red', 'blue'], fmt='.0f', 
                    cbar_kws={'label': 'Choice (1=Option1, 2=Option2)'},
                    xticklabels=op2_values, yticklabels=op1_values)
    else:
        # 유효한 선택이 없는 경우 빈 히트맵
        pivot_data = pd.DataFrame(index=op1_values, columns=op2_values)
        sns.heatmap(pivot_data, annot=True, cmap=['red', 'blue'], fmt='.0f',
                    cbar_kws={'label': 'Choice (1=Option1, 2=Option2)'},
                    xticklabels=op2_values, yticklabels=op1_values)
    
    plt.title(f'Choice Pattern for "{keyword}": op1_pos vs op2_pos', fontsize=16, pad=20)
    plt.xlabel('op2_pos Value', fontsize=12)
    plt.ylabel('op1_pos Value', fontsize=12)
    
    plt.tight_layout()
    
    # 히트맵 저장
    if output_path:
        filename = f"{output_path}_unified_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"💾 이미지 저장: {filename}")
    
    plt.close()  # 메모리에서 그래프 제거
    
    # extent 정보가 있다면 추가 분석
    if 'extent1' in df.columns and 'extent2' in df.columns:
        print(f"\n📊 Extent 정보 분석 - '{keyword}':")
        
        # extent1과 extent2의 패턴 분석
        extent_analysis = df.groupby(['op1_pos_num', 'op2_pos_num']).agg({
            'extent1': 'first',
            'extent2': 'first',
            'choice': 'first'
        }).reset_index()
        
        print("샘플 extent 패턴:")
        for _, row in extent_analysis.head(5).iterrows():
            print(f"  op1_pos={row['op1_pos_num']}, op2_pos={row['op2_pos_num']}: "
                  f"extent1={row['extent1']}, extent2={row['extent2']}, choice={row['choice']}")

def main():
    """메인 실행 함수"""
    print("🚀 밸런스 게임 결과 분석 시작")
    print("=" * 50)
    
    # 데이터 파일 경로 (절대 경로로 변경)
    data_file = "/home/chewon1227/careerpathway/data/perma/balance_game_results.jsonl"
    
    # 데이터 로드 및 분석
    analysis_data = analyze_balance_results(data_file)
    
    if analysis_data:
        # 시각화 생성
        create_visualization(analysis_data, output_path="balance_results_analysis")
        print("\n🎉 분석 완료!")
    else:
        print("\n❌ 분석할 데이터가 없습니다.")

if __name__ == "__main__":
    main()
