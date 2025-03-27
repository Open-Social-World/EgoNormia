import time
import random
from functools import wraps

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