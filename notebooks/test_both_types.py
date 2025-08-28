#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perma_analysis import *
import matplotlib.pyplot as plt

def test_both_question_types():
    """word와 factor 데이터 타입을 모두 테스트합니다."""
    
    print("=== WORD 데이터 테스트 ===")
    print("=" * 50)
    
    # WORD 데이터 테스트
    df_word = load_model_data('7b')
    win_matrix_word, total_wins_word = calculate_factor_wins(df_word, 'word')
    
    print("Win Matrix (WORD):")
    print(win_matrix_word)
    print("\nTotal Wins (WORD):")
    for factor, wins in total_wins_word.items():
        print(f"  {factor}: {wins}")
    
    # WORD 시각화 생성
    fig_word = create_perma_visualization('7b', 'word')
    plt.suptitle('WORD 데이터 시각화', fontsize=16, fontweight='bold')
    plt.show()
    
    print("\n" + "=" * 50)
    print("=== FACTOR 데이터 테스트 ===")
    print("=" * 50)
    
    # FACTOR 데이터 테스트
    win_matrix_factor, total_wins_factor = calculate_factor_wins(df_word, 'factor')
    
    print("Win Matrix (FACTOR):")
    print(win_matrix_factor)
    print("\nTotal Wins (FACTOR):")
    for factor, wins in total_wins_factor.items():
        print(f"  {factor}: {wins}")
    
    # FACTOR 시각화 생성
    fig_factor = create_perma_visualization('7b', 'factor')
    plt.suptitle('FACTOR 데이터 시각화', fontsize=16, fontweight='bold')
    plt.show()
    
    print("\n" + "=" * 50)
    print("=== 통합 시각화 테스트 ===")
    print("=" * 50)
    
    # 통합 시각화 테스트 (WORD)
    print("WORD 데이터 통합 시각화 생성 중...")
    combined_word = create_combined_visualization('word')
    plt.suptitle('WORD 데이터 통합 시각화', fontsize=16, fontweight='bold')
    plt.show()
    
    # 통합 시각화 테스트 (FACTOR)
    print("FACTOR 데이터 통합 시각화 생성 중...")
    combined_factor = create_combined_visualization('factor')
    plt.suptitle('FACTOR 데이터 통합 시각화', fontsize=16, fontweight='bold')
    plt.show()

if __name__ == "__main__":
    test_both_question_types()

