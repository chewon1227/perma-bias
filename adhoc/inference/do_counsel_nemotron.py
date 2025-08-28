from careerpathway.llm import get_results
from careerpathway.data import load_nemotron_persona
import os
from careerpathway.utils import get_random_name
import random
from tqdm import tqdm

def change_name(original_name: str, new_name: str, text: str):
    return text.replace(original_name, new_name)

def _normalize_sex_str(sex: str):
    if 'male' in sex.lower():
        return 'M'
    elif 'female' in sex.lower():
        return 'F'
    else:
        return random.choice(['M', 'F'])

def main(
    model_name_or_path : str = "Qwen/Qwen2.5-32B-Instruct",
    result_dir : str = "results/logs/nemotron_counsel",
    max_model_len : int = 8192,
    max_num_seqs : int = 50,
    gpu_memory_utilization : float = 0.90,
    country: str = 'us',
    
):
    os.makedirs(result_dir, exist_ok=True)
    gpt_model = 'gpt' in model_name_or_path or 'o1' in model_name_or_path
    data = load_nemotron_persona(sampling_num=10000)
    data = [{**item, 'country_name': get_random_name(country, _normalize_sex_str(item['sex']), name_type='Romanized Name')} for item in tqdm(data)]
    data = [
        {
            'inputs' : [
                change_name(item['name'], item['name'], item['country_name']),
                item['age'], 
                item['sex'], 
                change_name(item['education_level'], item['name'], item['country_name']), 
                change_name(item['skills_and_expertise'], item['name'], item['country_name']),
                change_name(item['hobbies_and_interests'], item['name'], item['country_name']),
                change_name(item['travel_persona'], item['name'], item['country_name']),
                change_name(item['marital_status'], item['name'], item['country_name'])
            ],
            'uuid' : item['uuid'],
            'country' : country,
            'name' : item['name'],
            'country_name' : item['country_name'],
        } for item in data
    ]
    print([item['country_name'] for item in data[:10]])

    prompt = """Based on the provided profile information, analyze the person's background, skills, interests, and personal characteristics to recommend 20 realistic and well-matched career options that would likely result in high job satisfaction and career success.

Please provide:
1. A brief analysis of the person's profile strengths, interests, and career-relevant characteristics
2. 20 specific career recommendations ranked by compatibility, including:
   - Job title and brief description
   - Why this career matches their profile (skills, interests, personality fit)
   - Realistic entry requirements and pathways
   - Expected job satisfaction factors for this person
   - Potential earning range and career growth prospects
3. For each recommendation, consider:
   - Alignment with their existing skills and expertise
   - Connection to their hobbies and interests
   - Compatibility with their lifestyle preferences and personal situation
   - Realistic accessibility given their education and background
   - Long-term career satisfaction and fulfillment potential

Focus on providing diverse, realistic career options that span different industries and career levels, prioritizing those with the highest likelihood of professional satisfaction and success for this specific individual.

Name: {0}
Age: {1}
Gender: {2}
Education: {3}
---------
SKills and Expertise: {4}
Hobbies and Interests: {5}
Travel Persona: {6}
Marital Status: {7}
"""
    results = get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=data,
        batch_size=1 if gpt_model else max_num_seqs,
        max_tokens=2000,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        save_path=f'{result_dir}/{model_name_or_path.replace("/", "_")}_{country}.jsonl'
    )
    
    return results


if __name__ == "__main__":
    import fire
    fire.Fire(main)

