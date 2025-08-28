import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_model_data(model_name):
    """모델 데이터를 로드하고 llm_choice 필드를 생성합니다."""
    original_path = f"data/final_perma/{model_name}.jsonl"
    reverse_path = f"data/final_perma/{model_name}-re.jsonl"
    
    df = pd.read_json(path_or_buf=original_path, lines=True)
    reversed_df = pd.read_json(path_or_buf=reverse_path, lines=True)
    
    # 노트북의 정확한 extract_choice 함수 사용
    df['llm_choice'] = df['llm_response'].apply(lambda x: extract_choice(x, reversed_mode=False))
    reversed_df['llm_choice'] = reversed_df['llm_response'].apply(lambda x: extract_choice(x, reversed_mode=True))
    
    # unknown을 NaN으로 변환
    df['llm_choice'] = df['llm_choice'].replace('unknown', np.nan)
    reversed_df['llm_choice'] = reversed_df['llm_choice'].replace('unknown', np.nan)
    
    # 숫자로 변환
    df['llm_choice'] = pd.to_numeric(df['llm_choice'], errors='coerce')
    reversed_df['llm_choice'] = pd.to_numeric(reversed_df['llm_choice'], errors='coerce')
    
    # 두 데이터프레임 합치기
    combined_df = pd.concat([df, reversed_df], ignore_index=True)
    
    return combined_df

def extract_numbers_from_description(description):
    """설명에서 숫자를 추출합니다."""
    if not description or pd.isna(description):
        return None
    
    try:
        # 숫자만 추출
        numbers = re.findall(r'\d+', str(description))
        if numbers:
            return int(numbers[0])  # 첫 번째 숫자 반환
        return None
    except Exception as e:
        print(f"Error extracting number from '{description}': {e}")
        return None

def analyze_perma_extent_battles(combined_df):
    """PERMA 조합별로 extent 기반 전투를 분석합니다."""
    # PERMA 요소들
    perma_factors = ['P', 'E', 'R', 'M', 'A']
    
    # 모든 가능한 PERMA 조합 생성
    perma_combinations = []
    for i in range(len(perma_factors)):
        for j in range(i + 1, len(perma_factors)):
            perma_combinations.append((perma_factors[i], perma_factors[j]))
    
    print(f"PERMA combinations: {perma_combinations}")
    
    # 각 조합별로 분석
    all_battle_results = {}
    
    for factor1, factor2 in perma_combinations:
        # 해당 조합의 reddit-com 데이터 필터링
        battle_data = combined_df[combined_df["question_id"] == "reddit-com"].copy()
        
        # reddit_type에서 해당 조합 찾기 (예: 'p-e', 'e-r' 등)
        factor1_lower = factor1.lower()
        factor2_lower = factor2.lower()
        
        # reddit_type 패턴 매칭
        reddit_patterns = [
            f"{factor1_lower}-{factor2_lower}",
            f"{factor2_lower}-{factor1_lower}"
        ]
        
        filtered_data = battle_data[battle_data["reddit_type"].isin(reddit_patterns)]
        
        print(f"{factor1} vs {factor2}: {len(filtered_data)} rows")
        
        if len(filtered_data) == 0:
            continue
        
        # extent 범위 파악
        factor1_extents = []
        factor2_extents = []
        
        for _, row in filtered_data.iterrows():
            op1_pos_num = extract_numbers_from_description(row.get("op1_pos", ""))
            op2_pos_num = extract_numbers_from_description(row.get("op2_pos", ""))
            
            if op1_pos_num is not None:
                factor1_extents.append(op1_pos_num)
            if op2_pos_num is not None:
                factor2_extents.append(op2_pos_num)
        
        if not factor1_extents or not factor2_extents:
            print(f"  No valid extent data found")
            continue
        
        factor1_range = (min(factor1_extents), max(factor1_extents))
        factor2_range = (min(factor2_extents), max(factor2_extents))
        
        print(f"  Factor1 ({factor1}) range: {factor1_range}")
        print(f"  Factor2 ({factor2}) range: {factor2_range}")
        
        # keyword별로 데이터 분리 (keyword는 op1_pos에서 추출)
        keyword_results = {}
        
        for _, row in filtered_data.iterrows():
            op1_pos_num = extract_numbers_from_description(row.get("op1_pos", ""))
            op2_pos_num = extract_numbers_from_description(row.get("op2_pos", ""))
            llm_choice = row.get('llm_choice')
            
            if pd.isna(llm_choice) or op1_pos_num is None or op2_pos_num is None:
                continue
            
            # keyword 필드 직접 사용
            keyword = row.get("keyword", "unknown")
            if not keyword or keyword == "unknown":
                # keyword가 없으면 op1_pos에서 숫자 제외한 부분 사용
                op1_pos_text = str(row.get("op1_pos", ""))
                keyword = re.sub(r'\d+', '', op1_pos_text).strip()
                if not keyword:
                    keyword = "unknown"
            
            # extent 조합을 키로 사용
            key = (op1_pos_num, op2_pos_num)
            
            if keyword not in keyword_results:
                keyword_results[keyword] = defaultdict(lambda: {'factor1_wins': 0, 'factor2_wins': 0})
            
            if llm_choice == 1:
                keyword_results[keyword][key]['factor1_wins'] += 1
            elif llm_choice == 2:
                keyword_results[keyword][key]['factor2_wins'] += 1
        
        # 전체 평균 결과 계산
        overall_results = defaultdict(lambda: {'factor1_wins': 0, 'factor2_wins': 0})
        for keyword_data in keyword_results.values():
            for key, data in keyword_data.items():
                overall_results[key]['factor1_wins'] += data['factor1_wins']
                overall_results[key]['factor2_wins'] += data['factor2_wins']
        
        all_battle_results[(factor1, factor2)] = {
            'keyword_results': keyword_results,
            'overall_results': overall_results,
            'factor1_range': factor1_range,
            'factor2_range': factor2_range
        }
        
        # 결과 요약 출력
        total_factor1_wins = sum(data['factor1_wins'] for data in overall_results.values())
        total_factor2_wins = sum(data['factor2_wins'] for data in overall_results.values())
        print(f"  {factor1}: {total_factor1_wins}, {factor2}: {total_factor2_wins}")
        print(f"  Keywords found: {list(keyword_results.keys())}")
    
    return all_battle_results

def create_perma_extent_barplot(battle_results, model_name, plot_type="overall"):
    """PERMA 조합별 extent 기반 전투 결과를 막대그래프로 표시합니다."""
    # PERMA 요소들
    perma_factors = ['P', 'E', 'R', 'M', 'A']
    
    # 모든 가능한 PERMA 조합 생성
    perma_combinations = []
    for i in range(len(perma_factors)):
        for j in range(i + 1, len(perma_factors)):
            perma_combinations.append((perma_factors[i], perma_factors[j]))
    
    # 10개의 subplot 생성 (2x5)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    plot_title = "전체 평균" if plot_type == "overall" else f"Keyword: {plot_type}"
    fig.suptitle(f'{model_name} - PERMA 조합별 Extent 기반 전투 결과 ({plot_title})', fontsize=16, fontweight='bold')
    
    # 각 PERMA 조합별로 subplot 생성
    for idx, (factor1, factor2) in enumerate(perma_combinations):
        if idx >= 10:  # 최대 10개 subplot
            break
            
        ax = axes[idx]
        
        # 해당 조합의 전투 결과
        if (factor1, factor2) in battle_results:
            result = battle_results[(factor1, factor2)]
            factor1_range = result['factor1_range']
            factor2_range = result['factor2_range']
            
            # plot_type에 따라 데이터 선택
            if plot_type == "overall":
                battle_data = result['overall_results']
            else:
                battle_data = result['keyword_results'].get(plot_type, {})
            
            # extent 값들 준비
            factor1_values = list(range(factor1_range[0], factor1_range[1] + 1))
            factor2_values = list(range(factor2_range[0], factor2_range[1] + 1))
            
            # 막대그래프 데이터 준비
            x_positions = []
            factor1_heights = []
            factor2_heights = []
            labels = []
            
            for factor1_val in factor1_values:
                for factor2_val in factor2_values:
                    key = (factor1_val, factor2_val)
                    if key in battle_data:
                        factor1_wins = battle_data[key]['factor1_wins']
                        factor2_wins = battle_data[key]['factor2_wins']
                        
                        if factor1_wins > 0 or factor2_wins > 0:  # 데이터가 있는 경우만
                            x_positions.append(len(x_positions))
                            factor1_heights.append(factor1_wins)
                            factor2_heights.append(factor2_wins)
                            labels.append(f'{factor1_val}-{factor2_val}')
            
            if x_positions:  # 데이터가 있는 경우만
                # 막대그래프 생성 (stacked bar)
                x = np.arange(len(x_positions))
                width = 0.8
                
                # factor1 선택 (파란색)
                bars1 = ax.bar(x, factor1_heights, width, label=factor1, color='lightblue', alpha=0.8)
                # factor2 선택 (빨간색) - factor1 위에 쌓기
                bars2 = ax.bar(x, factor2_heights, width, bottom=factor1_heights, label=factor2, color='lightcoral', alpha=0.8)
                
                # x축 레이블 설정
                ax.set_xlabel('Extent 조합')
                ax.set_ylabel('선택 횟수')
                ax.set_title(f'{factor1} vs {factor2}')
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.legend()
                
                # 막대 위에 숫자 표시
                for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
                    height1 = factor1_heights[i]
                    height2 = factor2_heights[i]
                    
                    if height1 > 0:
                        ax.text(bar1.get_x() + bar1.get_width()/2., height1/2, 
                               str(height1), ha='center', va='center', fontweight='bold')
                    if height2 > 0:
                        ax.text(bar2.get_x() + bar2.get_width()/2., height1 + height2/2, 
                               str(height2), ha='center', va='center', fontweight='bold')
                
                # 그리드 추가
                ax.grid(True, alpha=0.3)
                
            else:
                ax.set_title(f'{factor1} vs {factor2}\n(데이터 없음)')
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=14)
        else:
            ax.set_title(f'{factor1} vs {factor2}\n(데이터 없음)')
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=14)
    
    # 사용하지 않는 subplot 숨기기
    for idx in range(len(perma_combinations), 10):
        if idx < len(axes):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """메인 함수"""
    models = ['7b', '14b', '32b']
    
    for model in models:
        print(f"\n=== Processing {model} ===")
        
        try:
            # load_model_data 함수로 데이터 로드
            combined_df = load_model_data(model)
            
            # PERMA 조합별 extent 기반 전투 분석
            battle_results = analyze_perma_extent_battles(combined_df)
            
            if battle_results:
                # 1. 전체 평균 결과 막대그래프 생성
                fig = create_perma_extent_barplot(battle_results, model.upper(), "overall")
                plt.show()
                print(f"    Displayed {model} overall results")
                
                # 2. keyword별 결과 막대그래프 생성
                if battle_results:
                    # 첫 번째 조합에서 keyword 목록 가져오기
                    first_key = list(battle_results.keys())[0]
                    keywords = list(battle_results[first_key]['keyword_results'].keys())
                    
                    for keyword in keywords:
                        fig = create_perma_extent_barplot(battle_results, model.upper(), keyword)
                        plt.show()
                        print(f"    Displayed {model} {keyword} results")
                
            else:
                print(f"    No PERMA extent battle data found")
                
        except Exception as e:
            print(f"Error processing {model}: {e}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
