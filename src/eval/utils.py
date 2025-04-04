import time
from functools import wraps
import logging
import os
import csv



def backoff(max_retries=5, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        print(f"Max retries reached for {func.__name__}. Operation failed.")
                        raise e
                    
                    # Calculate the delay time using exponential backoff with randomization
                    delay = base_delay * (2 ** retries)
                    print(f"Retryable error during {func.__name__}: {e}. Retry {retries}/{max_retries}")
                    time.sleep(delay)
        return wrapper
    return decorator

def setup_logger(log_name: str = '', level: int = logging.DEBUG):

    # Make ../results directory if it doesn't exist
    if not os.path.exists(f'../results/{log_name}'):
        os.makedirs(f'../results/{log_name}')

    log_file = '../results/'+log_name+'/egonormia.log'

    name = "egonormia"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class ReasoningCache:
    def __init__(self, dir_name: str = ''):
        self.cache = []

        self.cachefile = '../results/'+dir_name+'/reasoning_cache.csv'

        # Create file if it doesn't exist
        if not os.path.exists(self.cachefile):
            with open(self.cachefile, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'task', 'reasoning'])

    def add_and_write(self, id: str, prompt: str, reasoning: str):

        task = None

        if 'choose ALL the sensible actions' in prompt:
            task = 'sensible'
        elif 'choose the single most normatively relevant or appropriate action' in prompt:
            task = 'best'
        elif 'choose the most normatively correct justification' in prompt:
            task = 'justification'

        self.cache.append([id, task, reasoning])

        # Write to file
        with open(self.cachefile, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([id, task, reasoning])
            f.flush()