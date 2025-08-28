from careerpathway.llm import get_results
from careerpathway.data import load_reddit_data
import os


def main(
    model_name_or_path : str = "Qwen/Qwen2.5-32B-Instruct",
    result_dir : str = "results/logs/reddit_counsel",
    max_model_len : int = 8192,
    max_num_seqs : int = 50,
    gpu_memory_utilization : float = 0.90,
):
    os.makedirs(result_dir, exist_ok=True)
    gpt_model = 'gpt' in model_name_or_path or 'o1' in model_name_or_path
    data = load_reddit_data(sampling_num=200 if gpt_model else None)
    prompt = """Please provide comprehensive career counseling based on the following situation. Whether the person is experiencing cognitive challenges (like career decision-making difficulties, planning issues, or skill development needs) or emotional challenges (like career-related stress, job satisfaction issues, or professional identity concerns), provide the most helpful and practical career guidance possible.

Please provide:
1. A thorough analysis of the person's career situation, identifying both cognitive and emotional aspects
2. A detailed career development plan that includes:
   - Immediate actionable steps they can take
   - Short-term goals (3-6 months)
   - Long-term career objectives (1-3 years)
   - Specific skills, qualifications, or experiences they should develop
   - Resources or tools that could help them
3. Practical strategies for:
   - Overcoming current career obstacles
   - Building professional confidence
   - Making informed career decisions
   - Managing career transitions
   - Maintaining work-life balance

Focus on providing clear, actionable advice that addresses both the practical and emotional aspects of their career development journey.


{0}
--------------------
{1}"""


    results = get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=data,
        batch_size=1 if gpt_model else max_num_seqs,
        max_tokens=2000,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        save_path=f'{result_dir}/{model_name_or_path.replace("/", "_")}.jsonl'
    )
    
    return results


if __name__ == "__main__":
    import fire
    fire.Fire(main)

