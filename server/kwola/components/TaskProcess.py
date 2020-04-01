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
