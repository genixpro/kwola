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

def autoretry(onFailure=None, maxAttempts=5, ignoreFailure=False, logRetries=True):
    def internalAutoretry(targetFunc):
        def retryFunction(*args, **kwargs):
            for attempt in range(maxAttempts):
                try:
                    return targetFunc(*args, **kwargs)
                except Exception as e:
                    if attempt == maxAttempts - 1:
                        if not ignoreFailure:
                            raise
                    else:
                        time.sleep(1.5 ** attempt + random.uniform(-0.5, 0.5))
                        if logRetries:
                            getLogger().info(f"Had to autoretry the function {targetFunc.__name__} due to the following exception: {traceback.format_exc()}")
                        if onFailure is not None:
                            onFailure(*args, **kwargs)
        return retryFunction
    return internalAutoretry
