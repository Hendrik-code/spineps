import time

import GPUtil
from TPTBox import Log_Type, No_Logger

logger = No_Logger()


def get_gpu(verbose: bool = False, maxLoad: float = 0.3, maxMemory: float = 0.4):
    GPUtil.showUtilization() if verbose else None
    deviceIDs = GPUtil.getAvailable(
        order="load",
        limit=4,
        maxLoad=maxLoad,
        maxMemory=maxMemory,
        includeNan=False,
        excludeID=[],
        excludeUUID=[],
    )
    return deviceIDs


def intersection(lst1, lst2):
    return set(lst1).intersection(lst2)


def get_free_gpus(blocked_gpus=None, maxLoad: float = 0.3, maxMemory: float = 0.4):
    # print("get_free_gpus")
    if blocked_gpus is None:
        blocked_gpus = {0: False, 1: False, 2: False, 3: False}
    cached_list = get_gpu(maxLoad=maxLoad, maxMemory=maxMemory)
    for i in range(15):
        time.sleep(0.25)
        cached_list = intersection(cached_list, get_gpu())
    # print("result:", list(cached_list))
    gpulist = [i for i in list(cached_list) if i not in blocked_gpus or blocked_gpus[i] == False]
    # print("result:", gpulist)
    return gpulist


def thread_print(fold, *text):
    global logger
    logger.print(f"Fold [{fold}]: ", *text)
