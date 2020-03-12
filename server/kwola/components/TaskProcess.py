import json

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
        dataStr = input()
        data = json.load(dataStr)
        result = self.targetFunc(**data)
        print(TaskProcess.resultStartString)
        print(json.dumps(result))
        print(TaskProcess.resultFinishString)
        exit(0)
