#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
