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


import sys
import json
from datetime import datetime

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
        print(datetime.now(), "TaskProcess: Waiting for input from stdin", flush=True)
        dataStr = sys.stdin.readline()
        data = json.loads(dataStr)
        print(datetime.now(), "Running process with following data:", flush=True)
        print(json.dumps(data, indent=4), flush=True)
        result = self.targetFunc(**data)
        print(TaskProcess.resultStartString, flush=True)
        print(json.dumps(result), flush=True)
        print(TaskProcess.resultFinishString, flush=True)
        exit(0)
