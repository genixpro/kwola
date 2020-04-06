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


import subprocess
import psutil
import atexit
import time
import json
import threading
from datetime import datetime
import os
from ..components.TaskProcess import TaskProcess


class ManagedTaskSubprocess:
    """
        This class is used to manage Kwola task subprocesses. These are subprocesses used in Kwola to due specific
        resource heavy tasks.

        They communicate with the master "manager" (this class) using a very simple JSON line-by-line communication
        scheme using just regular standard input / output.

        They will also be monitored with a timeout and such.
    """

    def __init__(self, args, data, timeout, config, logId):
        self.logFilePath = os.path.join(config.getKwolaUserDataDirectory("logs"), logId + "_log.txt")
        self.currentLogSize = 0

        processOutputFile = open(self.logFilePath, 'w')
        self.process = subprocess.Popen(args, stdout=processOutputFile, stderr=None, stdin=subprocess.PIPE)

        self.process.stdin.write(bytes(json.dumps(data) + "\n", "utf8"))
        self.process.stdin.flush()
        self.startTime = datetime.now()
        self.alive = True
        self.timeout = timeout

        self.output = ''

        self.monitorTimeoutProcess = threading.Thread(target=lambda: self.timeoutMonitoringThread())

    def __del__(self):
        self.process.terminate()

    def getLatestLogOutput(self):
        newSize = os.stat(self.logFilePath).st_size

        if newSize == self.currentLogSize:
            return None
        else:
            file = open(self.logFilePath, 'rt')
            file.seek(self.currentLogSize)
            data = file.read()
            self.currentLogSize += len(data)
            return data


    def gracefullyTerminateProcess(self):
        self.alive = False
        self.process.terminate()


    def hardKillProcess(self):
        self.alive = False
        try:
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            children.append(parent)
            for p in children:
                p.send_signal(9)
        except psutil.NoSuchProcess:
            pass


    def stopProcessBothMethods(self):
        # First send it the terminate signal and hope it exits gracefully
        if self.process.returncode is None:
            self.gracefullyTerminateProcess()
            time.sleep(3)

        # If it appears to still be running, give the entire tree of processes that this one touches a hard kill signal.
        # this should get the job done.
        if self.process.returncode is None:
            self.hardKillProcess()
            time.sleep(1)


    def extractResultFromOutput(self):
        if TaskProcess.resultStartString not in self.output or TaskProcess.resultFinishString not in self.output:
            print(datetime.now(), "Error! Unable to extract result from the subprocess. Its possible the subprocess may have died", flush=True)
            return None
        else:
            resultStart = self.output.index(TaskProcess.resultStartString)
            resultFinish = self.output.index(TaskProcess.resultFinishString)

            resultDataString = self.output[resultStart + len(TaskProcess.resultStartString) : resultFinish]
            result = json.loads(resultDataString)
            return result

    def doesOutputHaveExitString(self):
        # This is here to catch when a python exception happens in the sub-process but it did not fully die
        if "Traceback (most recent call last)" in self.output:
            return True

        if TaskProcess.resultFinishString in self.output:
            return True

        return False

    def waitForProcessResult(self):
        atexit.register(lambda: self.process.kill())

        waitBetweenStdoutUpdates = 0.2

        self.output = ''
        while self.process.returncode is None and (not self.doesOutputHaveExitString()) and self.alive:
            nextChars = self.getLatestLogOutput()

            if nextChars is not None:
                for nextChar in nextChars:
                    if nextChar == chr(127):
                        self.output = self.output[:-1]  # Erase the last character from the self.output.
                    else:
                        self.output += nextChar
                        print(nextChar, sep="", end="")
                print("", sep="", end="", flush=True)
            else:
                time.sleep(waitBetweenStdoutUpdates)

        print(datetime.now(), "Terminating task subprocess, task finished.", flush=True)
        self.alive = False
        self.stopProcessBothMethods()

        additionalOutput = self.getLatestLogOutput()
        if additionalOutput is not None:
            self.output += additionalOutput
            print(additionalOutput, sep="", end="", flush=True)

        result = self.extractResultFromOutput()
        print(datetime.now(), "Task Subprocess finished and gave back result", flush=True)
        print(json.dumps(result, indent=4), flush=True)

        return result



    def timeoutMonitoringThread(self):
        while self.alive:
            elapsedSeconds = (datetime.now() - self.startTime).total_seconds()
            if elapsedSeconds > self.timeout:
                print(datetime.now(), "Killing Process due to too much time elapsed", flush=True)
                self.stopProcessBothMethods()

            time.sleep(1)

