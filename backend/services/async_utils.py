import asyncio
import functools
import logging
import random
import time

logger = logging.getLogger(__name__)

async def retry_async(func, max_retries=3, base_delay=1, max_delay=10):
    """
    Retry an async function with exponential backoff.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        retries = 0
        while retries <= max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(f"Async retry failed after {max_retries} attempts: {e}")
                    raise
                
                delay = min(max_delay, base_delay * (2 ** (retries - 1)) + random.uniform(0, 1))
                logger.warning(f"Async call failed, retrying in {delay:.2f}s... (Attempt {retries}/{max_retries})")
                await asyncio.sleep(delay)
        return None
    return wrapper

async def run_with_timeout(coro, timeout_seconds, fallback_value=None):
    """
    Run a coroutine with a timeout and return fallback on timeout or error.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"Coroutine timed out after {timeout_seconds}s")
        return fallback_value
    except Exception as e:
        logger.error(f"Error in async task: {e}")
        return fallback_value
