from careerpathway.llm import get_results
from careerpathway.data import load_reddit_data
from careerpathway.utils import open_json, save_json
import os
import matplotlib.pyplot as plt
from termcolor import colored
import random
import itertools
import numpy as np
import json
import re

def parse_llm_output(text):
    """
    Parse JSON output from LLM response.
    
    Args:
        text (str): Raw text output from LLM
        
    Returns:
        dict: Parsed JSON data or None if parsing fails
    """
    if not text or not isinstance(text, str):
        print(f"Warning: Invalid input text: {text}")
        return None
        
    try:
        # Clean up the text
        text = text.strip()
        
        # Try to extract JSON from the text
        # Look for JSON object within ```json blocks first
        json_match = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_text = json_match.group(1).strip()
        else:
            # Look for JSON object directly - use non-greedy match and balance braces
            # Find the first opening brace and count braces to find matching closing brace
            start_idx = text.find('{')
            if start_idx == -1:
                # No JSON object found, try the entire text
                json_text = text
            else:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(text[start_idx:], start_idx):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if brace_count == 0:
                    json_text = text[start_idx:end_idx]
                else:
                    # Unbalanced braces, try the entire text
                    json_text = text
        
        # Clean up common JSON formatting issues
        json_text = json_text.strip()
        
        # Parse JSON
        data = json.loads(json_text)
        
        # Ensure it's a dictionary
        if not isinstance(data, dict):
            print(f"Warning: Parsed JSON is not a dictionary: {type(data)}")
            return None
        
        # Validate required fields
        required_fields = ['content', 'option_1', 'option_2', 'perma_1', 'perma_2']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"Warning: Missing required fields {missing_fields} in parsed output")
            print(f"Available fields: {list(data.keys())}")
            return None
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Attempted to parse: {json_text[:200]}...")
        return None
    except Exception as e:
        print(f"Unexpected error during parsing: {e}")
        print(f"Input text: {text[:200]}...")
        return None

def process_results(results):
    """
    Process and validate the results from LLM.
    
    Args:
        results (list): List of result dictionaries
        
    Returns:
        list: List of processed and validated results
    """
    if not results:
        print("Warning: No results to process")
        return []
    
    processed_results = []
    successful_parses = 0
    
    for i, result in enumerate(results):
        # Skip None results
        if result is None:
            print(f"Warning: Skipping None result at index {i}")
            continue
            
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            print(f"Warning: Result at index {i} is not a dictionary: {type(result)} - {result}")
            continue
        
        if 'output' not in result and 'result' not in result:
            print(f"Warning: No output/result found for result {i}")
            print(f"Available keys: {list(result.keys())}")
            continue
        
        # Handle both 'output' and 'result' keys for backward compatibility
        output_text = result.get('output') or result.get('result', '')
        
        parsed_data = parse_llm_output(output_text)
        if parsed_data:
            # Add original inputs for reference
            inputs = result.get('inputs', ['', '', ''])
            if not isinstance(inputs, list):
                print(f"Warning: Inputs is not a list for result {i}: {inputs}")
                inputs = ['', '', '']
            
            parsed_data['original_content'] = inputs[0] if len(inputs) > 0 else ''
            parsed_data['original_option_1'] = inputs[1] if len(inputs) > 1 else ''
            parsed_data['original_option_2'] = inputs[2] if len(inputs) > 2 else ''
            
            processed_results.append(parsed_data)
            successful_parses += 1
        else:
            print(f"Failed to parse result {i}")
            # Still keep the original data for debugging
            inputs = result.get('inputs', ['', '', ''])
            if not isinstance(inputs, list):
                inputs = ['', '', '']
                
            processed_results.append({
                'original_content': inputs[0] if len(inputs) > 0 else '',
                'original_option_1': inputs[1] if len(inputs) > 1 else '',
                'original_option_2': inputs[2] if len(inputs) > 2 else '',
                'raw_output': output_text,
                'parse_error': True
            })
    
    print(f"Successfully parsed {successful_parses}/{len(results)} results")
    return processed_results

def main(
    model_name_or_path: str = "gpt-4o-mini",
    only_parsing: bool = False,
    prompt_version: str = 'v1',
):
    data = open('data/tmp_perma_balance.txt', 'r').readlines()
    raw_data = []
    option_1, option_2 = None, None  # Initialize variables
    
    for line in data:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        if '💥' in line:
            try:
                parts = line.split('💥 ')[-1].split('vs')
                if len(parts) >= 2:
                    option_1 = parts[0].strip()
                    option_2 = parts[1].strip()
                else:
                    print(f"Warning: Invalid format in line with 💥: {line}")
                    continue
            except Exception as e:
                print(f"Error parsing options from line: {line}, Error: {e}")
                continue
        elif len(line) > 2 and option_1 is not None and option_2 is not None:
            raw_data.append({
                'inputs': [line, option_1, option_2],
                'content': line,
                'option_1': option_1,
                'option_2': option_2
            })
        elif len(line) > 2:
            print(f"Warning: Skipping line '{line}' - no options defined yet")
    
    print(f"Processed {len(raw_data)} items")

    
    if only_parsing:
        data = open_json(f'results/tmp_data/perma_balance_{model_name_or_path.replace("/", "_")}.jsonl')
        processed_results = process_results(data)
        save_json(processed_results, f'results/tmp_data/perma_balance_{model_name_or_path.replace("/", "_")}_processed.json')
        return

    if prompt_version == 'v1':
        prompt = """Given the following sentence that deals with a PERMA model-based dilemma, break it down into two options and express what values each represents.
If the sentence does not appropriately deal with a dilemma, rewrite the sentence by changing the details.
Please provide all responses in JSON format.
Make it all in english.

[Sentence]: {0}
[PERMA values]: {1}, {2}

Format:
{{
    "content": "sentence",
    "option_1": "option 1",
    "option_2": "option 2",
    "perma_1": "PERMA value of option 1",
    "perma_2": "PERMA value of option 2"
}}
"""

    results = get_results(
        model_name_or_path=model_name_or_path,
        prompt=prompt,
        data=raw_data,
        batch_size=10,
        max_tokens=2000,
        save_path=f'results/tmp_data/perma_balance_{model_name_or_path.replace("/", "_")}.jsonl'
    )

    processed_results = process_results(results)
    save_json(processed_results, f'results/tmp_data/perma_balance_{model_name_or_path.replace("/", "_")}_processed.json')



if __name__ == "__main__":
    import fire
    fire.Fire(main)

