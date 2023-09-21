#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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
