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

from ..config.logger import getLogger, setupLocalLogging
import logging
from datetime import datetime
import json
import sys
import os
import psutil
import time

class TaskProcess:
    """
        This class represents a task subprocess. This has the code that runs inside the sub-process which communicates
        upwards to the manager. See ManagedTaskSubprocess.
    """

    resultStartString = "======== TASK PROCESS RESULT START ========"
    resultFinishString = "======== TASK PROCESS RESULT END ========"

    def __init__(self, targetFunc):
        self.targetFunc = targetFunc


    def run(self):
        setupLocalLogging()
        getLogger().info(f"TaskProcess: Waiting for input from stdin")
        dataStr = sys.stdin.readline()
        data = json.loads(dataStr)
        getLogger().info(f"Running process with following data:\n{json.dumps({k:v for k,v in data.items() if k != 'config'}, indent=4)}")
        result = self.targetFunc(**data)
        print(TaskProcess.resultStartString + json.dumps(result) + TaskProcess.resultFinishString, flush=True)
        exit(0)
