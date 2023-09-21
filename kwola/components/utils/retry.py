#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


import time
import traceback
from ...config.logger import getLogger
import random
import threading

globalFailedRetryExceptions = {}

def autoretry(onFailure=None, maxAttempts=8, ignoreFailure=False, logRetries=True, exponentialBackOffBase=2, onFinalFailure=None):
    def internalAutoretry(targetFunc):
        def retryFunction(*args, **kwargs):
            global globalFailedRetryExceptions
            globalFailedRetryExceptions[threading.get_ident()] = set()

            stackMsg = "".join(traceback.format_stack()[:-1])
            for attempt in range(maxAttempts):
                try:
                    return targetFunc(*args, **kwargs)
                except Exception as e:
                    if attempt == maxAttempts - 1 or (id(e) in globalFailedRetryExceptions[threading.get_ident()]):
                        if onFinalFailure is not None:
                            onFinalFailure(*args, **kwargs)
                        if not ignoreFailure:
                            globalFailedRetryExceptions[threading.get_ident()].add(id(e))
                            raise
                    else:
                        time.sleep(exponentialBackOffBase ** (attempt + 1) * random.uniform(0.5, 1.5))
                        if logRetries:
                            getLogger().info(f"Had to autoretry the function {targetFunc.__name__} due to the following exception:\n{traceback.format_exc()}\nwhich was called from:\n{stackMsg}")
                        if onFailure is not None:
                            onFailure(*args, **kwargs)
        return retryFunction
    return internalAutoretry
