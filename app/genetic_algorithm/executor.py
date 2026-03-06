import asyncio
from concurrent.futures import ProcessPoolExecutor

# Matches your requirement of 5 simultaneous runs
ga_executor = ProcessPoolExecutor(max_workers=5)
_semaphore = asyncio.Semaphore(5)

def acquire():
    """
    Checks if available and consumes a slot immediately.
    """
    if _semaphore.locked():
        return False
    
    # Manually decrement the semaphore count to 'reserve' the slot
    # since we are calling this from a sync context in the router.
    _semaphore._value -= 1 
    return True

def release():
    _semaphore.release()