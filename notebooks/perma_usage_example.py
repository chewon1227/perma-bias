# PERMA Factor 분석 및 시각화 사용 예시

# 필요한 라이브러리 import
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perma_analysis import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def main():
    print("=== PERMA Factor 분석 시작 ===\n")
    
    # 1. WORD 데이터 분석
    print("1. WORD 데이터 분석")
    print("=" * 50)
    analyze_all_models('word')
    
    # 2. 통합 시각화 생성
    print("\n2. 통합 시각화 생성")
    print("=" * 50)
    combined_fig = create_combined_visualization('word')
    plt.show()
    
    # 3. 개별 모델 분석
    print("\n3. 7B 모델 개별 분석")
    print("=" * 50)
    df_7b = load_model_data('7b')
    win_matrix_7b, total_wins_7b = calculate_factor_wins(df_7b, 'word')
    
    print("7B 모델 - WORD 데이터 승수 행렬:")
    print(win_matrix_7b)
    print("\n7B 모델 - 각 factor 총 승수:")
    for factor, wins in total_wins_7b.items():
        print(f"{factor}: {wins}")
    
    # 4. 다른 question_type 분석
    print("\n4. FACTOR 데이터 분석")
    print("=" * 50)
    analyze_all_models('factor')
    
    print("\n5. REDDIT-POSITIVE 데이터 분석")
    print("=" * 50)
    analyze_all_models('reddit-positive')
    
    print("\n6. REDDIT-NEGATIVE 데이터 분석")
    print("=" * 50)
    analyze_all_models('reddit-negative')
    
    # 5. 데이터 구조 확인
    print("\n7. 데이터 구조 확인")
    print("=" * 50)
    df_sample = load_model_data('7b')
    print("데이터 컬럼:")
    print(df_sample.columns.tolist())
    print("\nquestion_id 값들:")
    print(df_sample['question_id'].unique())
    print("\n처음 5개 행:")
    print(df_sample[['question_id', 'factor1', 'factor2', 'llm_choice']].head())

if __name__ == "__main__":
    main()

