"""
Utility module for logging and measuring execution duration in the Smart PDF QA API application.
"""

import time, logging
from contextlib import asynccontextmanager
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

@asynccontextmanager
async def timing_block(name: str):
    """
    Asynchronously measures the execution time of a block of code.

    Args:
        name (str): The name of the operation being measured.
    """
    start_time = time.monotonic()
    try:
        logger.debug(f"Starting timing block for operation: {name}")
        yield 
    finally:
        end_time = time.monotonic()
        duration = end_time - start_time
        logger.info(f"Operation '{name}' executed in {duration:.4f} seconds.")


def log_duration(func):
    """
    Decorator for logging the execution duration of a function.

    Args:
        func: The function to measure.

    Returns:
        Function: The wrapped function with duration logging.
    """
    @wraps(func) 
    async def async_wrapper(*args, **kwargs):
        logger.debug(f"Starting async function: {func.__name__}")
        start_time = time.monotonic()
        result = await func(*args, **kwargs) 
        elapsed_time = (time.monotonic() - start_time) 
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} sec")
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger.debug(f"Starting sync function: {func.__name__}")
        start_time = time.monotonic()
        result = func(*args, **kwargs) 
        elapsed_time = (time.monotonic() - start_time)
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.2f} sec")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
    
def run_in_threadpool(func):
    """
    Asynchronously runs a function in a thread pool to avoid blocking the event loop.

    Args:
        func: The function to run in the thread pool.

    Returns:
        Function: The wrapped function that runs in the thread pool.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger.debug(f"Running function '{func.__name__}' in thread pool.")
        result = await asyncio.to_thread(func, *args, **kwargs)
        logger.info(f"Function '{func.__name__}' executed successfully in thread pool.")
        return result
    
    return wrapper