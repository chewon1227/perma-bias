import functools
from termcolor import colored
from typing import List, Dict, Any, Tuple
import graphviz


def log_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(colored(f"\n>>> Executing: {func.__name__}", 'green', attrs=['bold']))
        result = func(*args, **kwargs)
        print(colored(f"<<< Completed: {func.__name__}\n", 'green'))
        return result
    return wrapper

