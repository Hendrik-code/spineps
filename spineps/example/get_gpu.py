"""Helpers for selecting idle GPUs and thread-aware logging when running SPINEPS in parallel."""  # noqa: INP001

from __future__ import annotations

import time

import GPUtil
from TPTBox import No_Logger

logger = No_Logger()


def get_gpu(verbose: bool = False, max_load: float = 0.3, max_memory: float = 0.4):
    """Return the IDs of currently available GPUs below the given load and memory thresholds.

    Args:
        verbose (bool): If ``True``, print the current GPU utilization before querying.
        max_load (float): Maximum allowed compute load (0-1) for a GPU to count as available.
        max_memory (float): Maximum allowed memory usage (0-1) for a GPU to count as available.

    Returns:
        list[int]: Up to four available GPU IDs ordered by load.
    """
    GPUtil.showUtilization() if verbose else None
    device_ids = GPUtil.getAvailable(
        order="load",
        limit=4,
        maxLoad=max_load,
        maxMemory=max_memory,
        includeNan=False,
        excludeID=[],
        excludeUUID=[],
    )
    return device_ids


def intersection(lst1, lst2):
    """Return the set intersection of two iterables.

    Args:
        lst1: First iterable.
        lst2: Second iterable.

    Returns:
        set: Elements present in both ``lst1`` and ``lst2``.
    """
    return set(lst1).intersection(lst2)


def get_free_gpus(blocked_gpus=None, max_load: float = 0.3, max_memory: float = 0.4):
    """Poll the GPUs repeatedly and return those consistently free and not explicitly blocked.

    Availability is sampled 15 times (a short sleep between samples) and intersected so that only GPUs that stay
    idle across all samples are returned.

    Args:
        blocked_gpus (dict[int, bool] | None): Mapping of GPU ID to a blocked flag; a GPU is excluded when its flag
            is not ``False``. Defaults to ``{0: False, 1: False, 2: False, 3: False}``.
        max_load (float): Maximum allowed compute load (0-1) for the initial availability query.
        max_memory (float): Maximum allowed memory usage (0-1) for the initial availability query.

    Returns:
        list[int]: IDs of GPUs that are consistently available and not blocked.
    """
    # print("get_free_gpus")
    if blocked_gpus is None:
        blocked_gpus = {0: False, 1: False, 2: False, 3: False}
    cached_list = get_gpu(max_load=max_load, max_memory=max_memory)
    for _ in range(15):
        time.sleep(0.25)
        cached_list = intersection(cached_list, get_gpu())
    # print("result:", list(cached_list))
    gpulist = [i for i in list(cached_list) if i not in blocked_gpus or blocked_gpus[i] is False]
    # print("result:", gpulist)
    return gpulist


def thread_print(fold, *text):
    """Print a message prefixed with the fold identifier of the calling thread.

    Args:
        fold: Identifier of the fold/thread used as the message prefix.
        *text: Values to print after the prefix.
    """
    logger.print(f"Fold [{fold}]: ", *text)
