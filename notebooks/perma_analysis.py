import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def extract_choice(text, reversed_mode=False):
    """LLM 응답에서 선택된 옵션을 추출합니다."""
    text = str(text)
    idx1 = text.find("Option 1")
    idx2 = text.find("Option 2")

    if idx1 == -1 and idx2 == -1:
        choice = "unknown"
    elif idx1 != -1 and idx2 != -1:
        choice = "1" if idx1 < idx2 else "2"
    elif idx1 != -1:
        choice = "1"
    else:
        choice = "2"
    
    if reversed_mode:
        choice = "2" if choice == "1" else "1"
    
    return choice

def load_model_data(model_name):
    """모델 데이터를 로드하고 선택 결과를 처리합니다."""
    original_path = f"/home/chewon1227/careerpathway/data/final_perma/{model_name}.jsonl"
    reverse_path = f"/home/chewon1227/careerpathway/data/final_perma/{model_name}-re.jsonl"
    
    df = pd.read_json(path_or_buf=original_path, lines=True)
    reversed_df = pd.read_json(path_or_buf=reverse_path, lines=True)
    
    # 원본 데이터: Option 1이 factor1, Option 2가 factor2
    df['llm_choice'] = df['llm_response'].apply(lambda x: extract_choice(x, reversed_mode=False))
    
    # reverse 데이터: Option 1이 factor2, Option 2가 factor1 (반대로 처리)
    reversed_df['llm_choice'] = reversed_df['llm_response'].apply(lambda x: extract_choice(x, reversed_mode=True))
    
    # unknown을 NaN으로 변환
    df['llm_choice'] = df['llm_choice'].replace('unknown', np.nan)
    reversed_df['llm_choice'] = reversed_df['llm_choice'].replace('unknown', np.nan)
    
    # 숫자로 변환
    df['llm_choice'] = pd.to_numeric(df['llm_choice'], errors='coerce')
    reversed_df['llm_choice'] = pd.to_numeric(reversed_df['llm_choice'], errors='coerce')

    combined_df = pd.concat([df, reversed_df], ignore_index=True)
    
    return combined_df

def calculate_factor_wins(df, question_type):
    """
    특정 question_type에 대해 factor 간의 승수를 계산합니다.
    
    Args:
        df: 모델 데이터가 포함된 DataFrame
        question_type: 'word', 'factor', 'reddit-positive', 'reddit-negative' 중 하나
    
    Returns:
        win_matrix: 5x5 승수 행렬
        total_wins: 각 factor의 총 승수
    """
    # question_type에 해당하는 데이터만 필터링
    filtered_df = df[df['question_id'] == question_type].copy()
    
    # question_type에 따라 factors 매핑
    if question_type == 'word':
        # WORD 데이터: emotion, engagement, relationship, meaning, accomplishment
        factors = ['emotion', 'engagement', 'relationship', 'meaning', 'accomplishment']
        # PERMA 순서로 변환 (P, E, R, M, A)
        factor_order = ['emotion', 'engagement', 'relationship', 'meaning', 'accomplishment']
        display_factors = ['P', 'E', 'R', 'M', 'A']
    else:
        # FACTOR, REDDIT 데이터: P, E, R, M, A
        factors = ['P', 'E', 'R', 'M', 'A']
        factor_order = factors
        display_factors = factors
    
    # 승수 행렬 초기화 (5x5)
    win_matrix = pd.DataFrame(0, index=display_factors, columns=display_factors)
    
    # 각 factor 쌍에 대해 승수 계산
    for i, factor1 in enumerate(factors):
        for j, factor2 in enumerate(factors):
            if i != j:  # 자기 자신과는 비교하지 않음
                # factor1 vs factor2 비교 데이터 찾기
                mask = (filtered_df['factor1'] == factor1) & (filtered_df['factor2'] == factor2)
                comparison_data = filtered_df[mask]
                
                if len(comparison_data) > 0:
                    # Option 1이 이겼으면 factor1 승, Option 2가 이겼으면 factor2 승
                    wins_for_factor1 = (comparison_data['llm_choice'] == 1).sum()
                    wins_for_factor2 = (comparison_data['llm_choice'] == 2).sum()
                    
                    # display_factors 인덱스로 변환
                    if question_type == 'word':
                        # WORD 데이터의 경우 factor_order에서의 인덱스 찾기
                        idx1 = factor_order.index(factor1)
                        idx2 = factor_order.index(factor2)
                        display_factor1 = display_factors[idx1]
                        display_factor2 = display_factors[idx2]
                    else:
                        display_factor1 = factor1
                        display_factor2 = factor2
                    
                    # 승수 행렬에 기록
                    win_matrix.loc[display_factor1, display_factor2] = wins_for_factor1
                    win_matrix.loc[display_factor2, display_factor1] = wins_for_factor2
    
    # 각 factor의 총 승수 계산
    total_wins = {}
    for factor in display_factors:
        total_wins[factor] = win_matrix.loc[factor].sum()
    
    return win_matrix, total_wins

def create_perma_visualization(model_name, question_type):
    """
    특정 모델과 question_type에 대한 PERMA 시각화를 생성합니다.
    
    Args:
        model_name: 모델 이름 (예: '7b', '14b', '32b')
        question_type: 'word', 'factor', 'reddit-positive', 'reddit-negative' 중 하나
    
    Returns:
        fig: matplotlib figure 객체
    """
    # 데이터 로드
    df = load_model_data(model_name)
    
    # 승수 계산
    win_matrix, total_wins = calculate_factor_wins(df, question_type)
    
    # 시각화 생성 - 하나의 subplot만 사용
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 1. 가로 막대 그래프 (Total Wins) - 동적으로 factor 순서 결정
    if question_type == 'word':
        # WORD 데이터: P, E, R, M, A 순서
        factors = ['P', 'E', 'R', 'M', 'A']
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
    else:
        # FACTOR, REDDIT 데이터: P, E, R, M, A 순서
        factors = ['P', 'E', 'R', 'M', 'A']
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    # total_wins를 factors 순서대로 정렬
    wins_data = [total_wins[factor] for factor in factors]
    
    bars = ax.barh(factors, wins_data, color=colors)
    ax.set_xlim(0, 50)
    ax.set_xlabel('Total Wins')
    ax.set_ylabel('PERMA Factors')
    ax.set_title(f'{model_name.upper()} - {question_type.upper()}')
    
    # 막대 위에 값 표시
    for bar, win in zip(bars, wins_data):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{win}', va='center', fontweight='bold')
    
    # 2. Win Matrix 표를 막대 그래프 우하단에 배치
    # 표 위치 계산 (우하단)
    table_x = 0.65  # x 위치 (0~1, 오른쪽으로 갈수록 커짐)
    table_y = 0.15  # y 위치 (0~1, 아래쪽으로 갈수록 작아짐)
    table_width = 0.3  # 표 너비
    table_height = 0.4  # 표 높이
    
    # 표 데이터 준비 (동적으로 factor 순서 결정)
    table_data = []
    for factor1 in factors:
        row = []
        for factor2 in factors:
            if factor1 == factor2:
                row.append('-')
            else:
                row.append(str(win_matrix.loc[factor1, factor2]))
        table_data.append(row)
    
    # 표를 막대 그래프 위에 겹치기
    table = ax.table(cellText=table_data, 
                     rowLabels=factors,
                     colLabels=factors,
                     cellLoc='center',
                     loc='upper right',
                     bbox=[table_x, table_y, table_width, table_height])
    
    # 표 스타일링
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # 표 제목 추가
    ax.text(table_x + table_width/2, table_y + table_height + 0.02, 
            'Win Matrix', fontsize=12, fontweight='bold', 
            ha='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig

def analyze_all_models(question_type):
    """
    모든 모델에 대해 특정 question_type의 분석을 수행합니다.
    
    Args:
        question_type: 'word', 'factor', 'reddit-positive', 'reddit-negative' 중 하나
    """
    models = ['7b', '14b', '32b']
    
    for model in models:
        print(f"\n=== {model.upper()} - {question_type.upper()} 분석 ===")
        
        # 데이터 로드
        df = load_model_data(model)
        
        # 승수 계산
        win_matrix, total_wins = calculate_factor_wins(df, question_type)
        
        print("\nWin Matrix:")
        print(win_matrix)
        
        print("\nTotal Wins:")
        for factor, wins in total_wins.items():
            print(f"{factor}: {wins}")
        
        # 시각화 생성
        fig = create_perma_visualization(model, question_type)
        plt.show()

def create_combined_visualization(question_type):
    """
    모든 모델을 하나의 그림에 표시하는 통합 시각화를 생성합니다.
    
    Args:
        question_type: 'word', 'factor', 'reddit-positive', 'reddit-negative' 중 하나
    """
    models = ['7b', '14b', '32b']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 20))
    
    for i, model in enumerate(models):
        # 데이터 로드
        df = load_model_data(model)
        win_matrix, total_wins = calculate_factor_wins(df, question_type)
        
        # 가로 막대 그래프 - 동적으로 factor 순서 결정
        if question_type == 'word':
            # WORD 데이터: P, E, R, M, A 순서
            factors = ['P', 'E', 'R', 'M', 'A']
            colors = ['red', 'orange', 'yellow', 'green', 'blue']
        else:
            # FACTOR, REDDIT 데이터: P, E, R, M, A 순서
            factors = ['P', 'E', 'R', 'M', 'A']
            colors = ['red', 'orange', 'yellow', 'green', 'blue']
        
        wins_data = [total_wins[factor] for factor in factors]
        
        bars = axes[i].barh(factors, wins_data, color=colors)
        axes[i].set_xlim(0, 50)
        axes[i].set_xlabel('Total Wins')
        axes[i].set_ylabel('PERMA Factors')
        axes[i].set_title(f'{model.upper()} - {question_type.upper()}')
        
        # 막대 위에 값 표시
        for bar, win in zip(bars, wins_data):
            axes[i].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{win}', va='center', fontweight='bold')
        
        # Win Matrix 표를 막대 그래프 우하단에 배치
        # 표 위치 계산 (우하단)
        table_x = 0.65  # x 위치 (0~1, 오른쪽으로 갈수록 커짐)
        table_y = 0.15  # y 위치 (0~1, 아래쪽으로 갈수록 작아짐)
        table_width = 0.3  # 표 너비
        table_height = 0.4  # 표 높이
        
        table_data = []
        for factor1 in factors:
            row = []
            for factor2 in factors:
                if factor1 == factor2:
                    row.append('-')
                else:
                    row.append(str(win_matrix.loc[factor1, factor2]))
            table_data.append(row)
        
        # 표를 막대 그래프 위에 겹치기
        table = axes[i].table(cellText=table_data, 
                            rowLabels=factors,
                            colLabels=factors,
                            cellLoc='center',
                            loc='upper right',
                            bbox=[table_x, table_y, table_width, table_height])
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        
        # 표 제목 추가
        axes[i].text(table_x + table_width/2, table_y + table_height + 0.02, 
                    'Win Matrix', fontsize=10, fontweight='bold', 
                    ha='center', transform=axes[i].transAxes)
    
    plt.tight_layout()
    return fig

# 사용 예시
if __name__ == "__main__":
    # 특정 question_type 분석
    print("=== WORD 데이터 분석 ===")
    analyze_all_models('word')
    
    # 통합 시각화 생성
    print("\n=== 통합 시각화 생성 ===")
    combined_fig = create_combined_visualization('word')
    plt.show()
