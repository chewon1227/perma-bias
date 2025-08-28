from careerpathway.llm import get_results
from careerpathway.data import load_reddit_data
from careerpathway.utils import open_json, save_json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nrclex import NRCLex


def main(
    model_name_or_path : str = "Qwen/Qwen2.5-32B-Instruct",
    only_parsing : bool = False,
    parsing_type: int = 1,
    colors : list[str] = ['black', '#4BDBC5', 'lightgray'],
    ):
    if only_parsing:
        results = open_json(f"results/tmp_data/reddit_data_type_o1_200.jsonl")[-200:]
        valid_keys = [
            'lack of motivation', 'indecisiveness', 'dysfunctional beliefs', 'lack of information about the self', 'lack of information about occupations', 
            'lack of information about ways to make a decision', 'unreliable information', 'internal conflicts', 'external conflicts', 'difficulty implementing a choice', 
            'anxiety about the future', 'depression or hopelessness', 'low self-efficacy', 'low self-worth', 'emotional dysregulation', 'avoidance or denial', 'fear of failure or judgment', 'overwhelm or burnout']
        
        if parsing_type == 1:
            cognitive = 0
            emotional = 0
            no_type = 0
            for r in results:
                type = r['result'].split("\n")[0].lower().strip()
                # type = r['result'].split("\n\n")[0][-11:].strip(")").strip(".").strip().lower()
                print(type)
                print('-'*100)
                if 'both' in type:
                    cognitive += 1
                    emotional += 1
                elif 'cognitive' in type:
                    cognitive += 1
                elif 'emotional' in type:
                    emotional += 1
                else:
                    no_type += 1
            # draw pie chart
            fontsize = 20
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Define colors and text colors
            text_colors = ['white', 'black', 'black']
            sizes = [cognitive, emotional, no_type]
            labels = ['Cognitive', 'Emotional', 'N/A']
            
            # Create pie chart without labels (we'll add them manually)
            wedges, _ = ax.pie(
                sizes, 
                colors=colors,
                startangle=90,
                )
            
            # Add custom labels with individual distances and center alignment
            distances = [0.55, 0.5, 1.1]  # Different label distances for each slice
            
            for i, wedge in enumerate(wedges):
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = distances[i] * np.cos(np.deg2rad(angle))
                y = distances[i] * np.sin(np.deg2rad(angle))
                
                label = f"{labels[i]}\n({sizes[i]})"
                ax.text(x, y, label, ha='center', va='center', 
                       fontsize=fontsize, color=text_colors[i])
            
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('results/plots/reddit_data_type_pie_chart.png', bbox_inches='tight')
        
        elif parsing_type == 2:
            def _make_label(key):
                if 'lack of information about ways to make a decision' in key:
                    return 'Lack of information (Ways)'
                elif 'lack of information about occupations' in key:
                    return 'Lack of information (Occupations)'
                elif 'lack of information about the self' in key:
                    return 'Lack of information (Self)'
                return key.capitalize()
            
            result_cnt = defaultdict(int)
            for r in results:
                result = r['result'].lower()
                print(r['prompt'].split("Be concise, don't include any other text except the type of difficulty.\n\n")[-1])
                print('-'*30)
                print(result)
                print('-'*100)
                for key in valid_keys:
                    if key in result:
                        result_cnt[key] += 1
            # sorting and bar plot
            sorted_result_cnt = sorted(result_cnt.items(), key=lambda x: x[1], reverse=True)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            for i, (key, cnt) in enumerate(sorted_result_cnt):
                cog_type = valid_keys.index(key) < 10
                ax.bar(i, cnt, color=colors[0] if cog_type else colors[1])
                ax.text(i, cnt / 2, cnt, ha='center', va='center', fontsize=15, color='white' if cog_type else 'black')
                
            # edge 삭제
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.yticks([])
            
            # Set xticks with proper alignment
            plt.xticks(
                range(len(sorted_result_cnt)), 
                [_make_label(x[0])
                 for x in sorted_result_cnt], 
                rotation=30,
                fontsize=15, 
                ha='right',
                va='top'
            )
            plt.savefig('results/plots/reddit_data_type_barplot.png', bbox_inches='tight')

        elif parsing_type == 3:
            # Separate cognitive and emotional difficulties
            cognitive_keys = valid_keys[:10]  # First 10 are cognitive
            emotional_keys = valid_keys[10:]  # Last 8 are emotional
            
            # Create co-occurrence matrix
            co_occurrence = np.zeros((len(emotional_keys), len(cognitive_keys)))
            
            for r in results:
                result = r['result'].lower()
                
                # Find which cognitive and emotional difficulties appear in this result
                cog_present = [i for i, key in enumerate(cognitive_keys) if key in result]
                emo_present = [i for i, key in enumerate(emotional_keys) if key in result]
                
                # Increment co-occurrence for all combinations
                for emo_idx in emo_present:
                    for cog_idx in cog_present:
                        co_occurrence[emo_idx, cog_idx] += 1
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 7))
            fontsize = 15
            
            # Create labels for better readability
            def _make_short_label(key):
                if 'lack of information about ways to make a decision' in key:
                    return 'Info (Ways)'
                elif 'lack of information about occupations' in key:
                    return 'Info (Occupations)'
                elif 'lack of information about the self' in key:
                    return 'Info (Self)'
                elif 'lack of motivation' in key:
                    return 'Motivation'
                elif 'indecisiveness' in key:
                    return 'Indecisiveness'
                elif 'dysfunctional beliefs' in key:
                    return 'Dysfunctional Beliefs'
                elif 'unreliable information' in key:
                    return 'Unreliable Info'
                elif 'internal conflicts' in key:
                    return 'Internal Conflicts'
                elif 'external conflicts' in key:
                    return 'External Conflicts'
                elif 'difficulty implementing' in key:
                    return 'Implementation'
                elif 'anxiety about the future' in key:
                    return 'Future Anxiety'
                elif 'depression or hopelessness' in key:
                    return 'Depression'
                elif 'low self-efficacy' in key:
                    return 'Low Self-Efficacy'
                elif 'low self-worth' in key:
                    return 'Low Self-Worth'
                elif 'emotional dysregulation' in key:
                    return 'Dysregulation'
                elif 'avoidance or denial' in key:
                    return 'Avoidance'
                elif 'fear of failure' in key:
                    return 'Fear of Failure'
                elif 'overwhelm or burnout' in key:
                    return 'Overwhelm'
                return key.capitalize()
            
            cog_labels = [_make_short_label(key) for key in cognitive_keys]
            emo_labels = [_make_short_label(key) for key in emotional_keys]
            
            # Sort rows and columns to put high values in top-left
            # Sort emotional difficulties (rows) by their total co-occurrence
            emo_sums = co_occurrence.sum(axis=1)
            emo_sort_idx = np.argsort(emo_sums)[::-1]  # Descending order
            
            # Sort cognitive difficulties (columns) by their total co-occurrence
            cog_sums = co_occurrence.sum(axis=0)
            cog_sort_idx = np.argsort(cog_sums)[::-1]  # Descending order
            
            # Reorder matrix, labels
            co_occurrence_sorted = co_occurrence[emo_sort_idx, :][:, cog_sort_idx]
            emo_labels_sorted = [emo_labels[i] for i in emo_sort_idx]
            cog_labels_sorted = [cog_labels[i] for i in cog_sort_idx]
            
            # Create heatmap
            im = ax.imshow(co_occurrence_sorted, cmap='Blues', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(cog_labels_sorted)))
            ax.set_yticks(range(len(emo_labels_sorted)))
            ax.set_xticklabels(cog_labels_sorted, rotation=45, ha='right', fontsize=fontsize)
            ax.set_yticklabels(emo_labels_sorted, fontsize=fontsize)
            
            # Add text annotations
            for i in range(len(emo_labels_sorted)):
                for j in range(len(cog_labels_sorted)):
                    if co_occurrence_sorted[i, j] > 0:
                        text = ax.text(j, i, int(co_occurrence_sorted[i, j]),
                                     ha="center", va="center", color="white" if co_occurrence_sorted[i, j] > co_occurrence_sorted.max()/2 else "black",
                                     fontsize=fontsize-2)
            
            # Labels and title
            ax.set_xlabel('Cognitive Difficulties', fontsize=fontsize, fontweight='bold')
            ax.set_ylabel('Emotional Difficulties', fontsize=fontsize, fontweight='bold')
            ax.set_title('Co-occurrence of Cognitive and Emotional Difficulties', fontsize=fontsize, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('results/plots/reddit_data_type_heatmap.png', bbox_inches='tight', dpi=300)

        elif parsing_type == 4:


            text = "I feel sad and anxious about my future."
            tokens = word_tokenize(text)

            emotions = []
            for token in tokens:
                emo = NRCLex(token).raw_emotion_scores
                if emo:
                    emotions.append((token, emo))

            print(emotions)

        return
    
    
    os.makedirs('results/tmp_data', exist_ok=True)
    data = load_reddit_data(sampling_num=200 if 'gpt' in model_name_or_path or 'o1' in model_name_or_path else None)
    
    prompt = """
You are a psychologically informed agent. Please analyze the following career concern and make a three-step classification and diagnosis.

1. Determine the **primary type of difficulty** the person is facing.
If the person is facing both cognitive and emotional difficulties, choose **BOTH**.
- COGNITIVE
- EMOTIONAL
- BOTH

2. If **COGNITIVE** is involved, **select all that apply** from the following 10 specific difficulties (based on CDDQ framework):
- Lack of Motivation
- Indecisiveness
- Dysfunctional Beliefs
- Lack of Information About the Self
- Lack of Information About Occupations
- Lack of Information About Ways to Make a Decision
- Unreliable Information
- Internal Conflicts
- External Conflicts
- Difficulty Implementing a Choice

3. If **EMOTIONAL** is involved, **select all that apply** from the following:
- Anxiety about the future
- Depression or hopelessness
- Low self-efficacy
- Low self-worth
- Emotional dysregulation
- Avoidance or denial
- Fear of failure or judgment
- Overwhelm or burnout

Be concise, don't include any other text except the type of difficulty.

[Title]: {0}
[Content]: {1}
"""


    results = get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=data,
        batch_size=1,
        max_tokens=2000,
        save_path=f'results/tmp_data/reddit_data_type_{model_name_or_path.replace("/", "_")}_{len(data)}.jsonl'
    )
    
    return results


if __name__ == "__main__":
    import fire
    fire.Fire(main)

