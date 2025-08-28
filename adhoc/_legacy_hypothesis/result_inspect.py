import fire
import colorama
from mylmeval import open_json
from typing import List, Optional

def inspect_jsonl(
    data_path: str='/home/iyy1112/workspace/Career-Pathway/results/mcts_value_model/tmp_CohereForAI_aya-expanse-32b.jsonl',
    n: int = 10,
    columns: Optional[List[str]] = None,
    colors: Optional[List[str]] = None
) -> None:
    # Initialize colorama
    colorama.init()
    
    # Set defaults
    if columns is None:
        columns = ['prompt', 'result']
    if colors is None:
        colors = ['YELLOW', 'CYAN', 'MAGENTA', 'BLUE', 'RED', 'GREEN','WHITE']
    
    # Load data
    data = open_json(data_path)
    n = min(n, len(data))
    
    # Create color cycle
    color_cycle = [getattr(colorama.Fore, color) for color in colors]
    
    # Print examples
    for i in range(n):
        print(f"\n-------------\nExample {i+1}\n-------------")
        for col, color in zip(columns, color_cycle):
            print(color + f"{col}:")
            print(data[i].get(col, "N/A"))

if __name__ == "__main__":
    fire.Fire(inspect_jsonl)