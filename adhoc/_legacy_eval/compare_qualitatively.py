import evaluate
import pandas as pd
import re
from get_score import meteor_batch


def compare_qualitatively(file_path_gpt4, file_path_llama):
    """
    Compare GPT-4 and Llama-2-7b responses, printing cases where GPT-4 performs worse.
    
    Args:
        file_path_gpt4: Path to GPT-4 results file
        file_path_llama: Path to Llama-2-7b results file
    """
    # Load the data
    df_gpt4 = pd.read_json(file_path_gpt4, lines=True)
    df_llama = pd.read_json(file_path_llama, lines=True)
    
    # Add task information
    df_gpt4['task'] = df_gpt4['metadata'].apply(lambda x: x['task'])
    df_llama['task'] = df_llama['metadata'].apply(lambda x: x['task'])
    avg_length_gpt4 = df_gpt4['result'].apply(lambda x: len(x)).mean()
    avg_length_llama = df_llama['result'].apply(lambda x: len(x)).mean()
    print(f"Average response length (GPT-4): {avg_length_gpt4}")
    print(f"Average response length (Llama): {avg_length_llama}")
    
    
    meteor = evaluate.load('meteor')
    
    # Process each task type
    for task in ['requirement', 'description']:
        print(f"\nAnalyzing {task} responses...")
        
        # Filter data for current task
        gpt4_task = df_gpt4[df_gpt4['task'] == task]
        llama_task = df_llama[df_llama['task'] == task]
        
        # Process in batches
        batch_size = 1
        
        if task == 'salary':
            # Special handling for salary task
            for i, (gpt4_row, llama_row) in enumerate(zip(gpt4_task.itertuples(), llama_task.itertuples())):
                try:
                    # Process GPT-4 response
                    gpt4_nums = [int(x) for x in re.findall(r'\d+', gpt4_row.result.replace(',', ''))]
                    gpt4_val = gpt4_nums[0] if len(gpt4_nums) == 1 else sum(gpt4_nums[:2])/2
                    
                    # Process Llama response
                    llama_nums = [int(x) for x in re.findall(r'\d+', llama_row.result.replace(',', ''))]
                    llama_val = llama_nums[0] if len(llama_nums) == 1 else sum(llama_nums[:2])/2
                    
                    # Get reference value
                    reference_num = int(re.findall(r'\d+', str(gpt4_row.groundtruth))[0])
                    
                    # Calculate scores (squared difference)
                    gpt4_score = (float(gpt4_val) - reference_num) ** 2
                    llama_score = (float(llama_val) - reference_num) ** 2
                    
                    # Compare scores (for salary, lower is better)
                    if gpt4_score > llama_score:
                        print(f"\nFound worse GPT-4 salary prediction:")
                        print(f"--------\nGround truth: ${reference_num}")
                        print(f"--------\nGPT-4 prediction: ${gpt4_val} (diff: {gpt4_score})")
                        print(f"--------\nLlama prediction: ${llama_val} (diff: {llama_score})")
                        print(f"--------\nOriginal query: {gpt4_row.prompt}")
                except:
                    continue
        else:
            # Handle requirement and description tasks
            for i in range(0, len(gpt4_task), batch_size):
                # Get batches
                gpt4_batch = gpt4_task.iloc[i:i + batch_size]
                llama_batch = llama_task.iloc[i:i + batch_size]
                
                # Calculate METEOR scores
                gpt4_score = meteor_batch(gpt4_batch['result'].tolist(), 
                                       gpt4_batch['groundtruth'].tolist(), 
                                       meteor)
                llama_score = meteor_batch(llama_batch['result'].tolist(), 
                                        llama_batch['groundtruth'].tolist(), 
                                        meteor)
                
                # Compare scores (for METEOR, higher is better)
                if gpt4_score < llama_score:
                    print(f"\n--------\nFound worse GPT-4 {task}:")
                    print(f"--------\nGPT-4 METEOR score: {gpt4_score}")
                    print(f"--------\nLlama METEOR score: {llama_score}")
                    # Print a sample from this batch
                    sample_idx = 0
                    print(f"--------\nSample query: {gpt4_batch.iloc[sample_idx]['prompt']}")
                    print(f"--------\nGround truth: {gpt4_batch.iloc[sample_idx]['groundtruth']}")
                    print(f"--------\nGPT-4 response: {gpt4_batch.iloc[sample_idx]['result']}")
                    print(f"--------\nLlama response: {llama_batch.iloc[sample_idx]['result']}")


if __name__ == '__main__':
    gpt4_file = "results/eval_truthfulness/gpt-4o-mini.jsonl"
    llama_file = "results/eval_truthfulness/meta-llama_Llama-2-7b-chat-hf.jsonl"
    compare_qualitatively(gpt4_file, llama_file)