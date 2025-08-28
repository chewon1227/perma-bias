import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

def load_names_dict(
    name_type: str = 'Localized Name',
    file_dir: str = '../popular-names-by-country-dataset/common-forenames-by-country.csv'
    ): # Localized Name or Romanized Name
    
    country_names = {}
    df2 = pd.read_csv(file_dir)
    for country in df2['Country'].unique():
        F = df2[(df2['Country'] == country) & (df2['Gender'] == 'F')][name_type].tolist()
        M = df2[(df2['Country'] == country) & (df2['Gender'] == 'M')][name_type].tolist()
        if len(F) > 10 and len(M) > 10:
            country_names[country.lower()] = {'F': F, 'M': M}
    
    if name_type == 'Localized Name':
        country_names['cn'] = {'M' : ['伟', '军', '毅', '刚', '浩', '明', '杰', '峰', '磊', '涛'], 'F' : ['娜', '丽', '霞', '娟', '艳', '红', '敏', '英', '梅', '兰']}
        country_names['kr'] = {'M' : ['민수', '종호', '승훈', '지훈', '동현', '영호', '재우', '성진', '민호', '태영'], 'F' : ['지영', '민정', '수진', '은정', '혜진', '미정', '영희', '지혜', '은영', '미숙']}

    elif name_type == 'Romanized Name':
        country_names['cn'] = {'M' : ['Wei', 'Jun', 'Yi', 'Gang', 'Hao', 'Ming', 'Jie', 'Feng', 'Lei', 'Tao'], 'F' : ['Na', 'Li', 'Xia', 'Juan', 'Yan', 'Hong', 'Min', 'Ying', 'Mei', 'Lan']}
        country_names['kr'] = {'M' : ['Minsoo', 'Jongho', 'Seunghun', 'Jihoon', 'Donghyun', 'Youngho', 'Jaewoo', 'Sungjin', 'Minho', 'Taeyoung'], 'F' : ['Jiyoung', 'Minjung', 'Soojin', 'Eunjung', 'Hyejin', 'Mijung', 'Younghee', 'Jihye', 'Eunyoung', 'Misook']}

    return country_names


def get_random_name(nation: str, sex: str, name_type: str = 'Localized Name') -> str:
    country_names = load_names_dict(name_type=name_type)
    return np.random.choice(country_names[nation.lower()][sex.upper()])
    

def open_json(file_path):
    if file_path.endswith(".json"):
        with open(file_path, encoding='utf-8') as f:
            data = json.load(f)
        return data
    elif file_path.endswith(".jsonl"):
        data = []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

def save_json(data: List[Dict], path: str, mode: str = 'w'):
    if path.endswith(".json"):
        with open(path, mode, encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif path.endswith(".jsonl"):
        with open(path, mode, encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            
def load_api_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load API configuration from yaml file.
    
    Args:
        config_path (str, optional): Path to the config file. 
            Defaults to 'configs/api.yaml' relative to project root.
    
    Returns:
        Dict[str, Any]: Dictionary containing API configurations
    """
    if config_path is None:
        # Get the project root directory (assuming utils.py is in src/careerpathway)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "api.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def set_api_keys(config: Dict[str, Any] = None) -> None:
    """
    Set API keys as environment variables.
    
    Args:
        config (Dict[str, Any], optional): API configuration dictionary.
            If None, will load from default config file.
    """
    if config is None:
        config = load_api_config()
    
    # Set OpenAI API key
    if 'openai' in config and 'api_key' in config['openai']:
        os.environ['OPENAI_API_KEY'] = config['openai']['api_key']
        print(f"OpenAI API key set from config file")
    
    # Set Reddit API credentials
    if 'reddit' in config:
        reddit_config = config['reddit']
        if 'client_id' in reddit_config:
            os.environ['REDDIT_CLIENT_ID'] = reddit_config['client_id']
        if 'client_secret' in reddit_config:
            os.environ['REDDIT_CLIENT_SECRET'] = reddit_config['client_secret']
        if 'user_agent' in reddit_config:
            os.environ['REDDIT_USER_AGENT'] = reddit_config['user_agent']
        print(f"Reddit API credentials set from config file")
    
    # Add more API key settings as needed

def get_api_key(service: str, key_name: str) -> str:
    """
    Get API key from environment variables.
    
    Args:
        service (str): Service name (e.g., 'openai', 'reddit')
        key_name (str): Key name (e.g., 'api_key', 'client_id')
    
    Returns:
        str: API key value
    
    Raises:
        KeyError: If the requested API key is not found
    """
    env_var_name = f"{service.upper()}_{key_name.upper()}"
    if env_var_name not in os.environ:
        raise KeyError(f"API key {env_var_name} not found in environment variables")
    return os.environ[env_var_name]

# Example usage:
# config = load_api_config()
# set_api_keys(config)
# openai_key = get_api_key('openai', 'api_key')