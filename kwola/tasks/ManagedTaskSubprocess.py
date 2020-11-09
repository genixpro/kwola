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
from ..tasks.TaskProcess import TaskProcess
from datetime import datetime
import atexit
import json
import os
import psutil
import subprocess
import threading
import time
import tempfile
import sys


class ManagedTaskSubprocess:
    """
        This class is used to manage Kwola task subprocesses. These are subprocesses used in Kwola to due specific
        resource heavy tasks.

        They communicate with the master "manager" (this class) using a very simple JSON line-by-line communication
        scheme using just regular standard input / output.

        They will also be monitored with a timeout and such.
    """

    def __init__(self, args, data, timeout, config, logId):
        if logId is not None:
            self.logFilePath = os.path.join(config.getKwolaUserDataDirectory("logs"), logId + "_log.txt")
        else:
            self.logFilePath = tempfile.mktemp()
        self.currentLogSize = 0

        self.args = args
        self.data = data

        self.processOutputFile = open(self.logFilePath, 'w')
        self.process = None
        self.startTime = datetime.now()
        self.alive = True
        self.timeout = timeout

        self.output = ''

        self.monitorTimeoutProcess = None
        self.monitorOutputProcess = None

    def start(self):
        self.process = subprocess.Popen(self.args, stdout=self.processOutputFile, stderr=sys.stderr, stdin=subprocess.PIPE)

        atexit.register(lambda: self.process.terminate())

        self.process.stdin.write(bytes(json.dumps(self.data) + "\n", "utf8"))
        self.process.stdin.flush()
        self.process.stdin.close()

        self.monitorTimeoutProcess = threading.Thread(target=lambda: self.timeoutMonitoringThread(), daemon=True)
        self.monitorOutputProcess = threading.Thread(target=lambda: self.outputMonitoringThread(), daemon=True)


        self.monitorTimeoutProcess.start()
        self.monitorOutputProcess.start()


    def getLatestLogOutput(self):
        newSize = os.stat(self.logFilePath).st_size

        if newSize == self.currentLogSize:
            return None
        else:
            file = open(self.logFilePath, 'rt')
            file.seek(self.currentLogSize)
            data = file.read()
            self.currentLogSize += len(data)
            file.close()
            return data


    def gracefullyTerminateProcess(self, processes=None):
        self.alive = False
        if processes is None:
            processes = self.getAllChildProcesses()
        for p in processes:
            try:
                p.terminate()
                # getLogger().info(f"Terminated child process {p.pid}")
            except psutil.NoSuchProcess:
                pass


    def hardKillProcess(self, processes=None):
        self.alive = False
        if processes is None:
            processes = self.getAllChildProcesses()
        for p in processes:
            try:
                p.send_signal(9)
                # getLogger().info(f"Killed (signal 9) child process {p.pid}")
            except psutil.NoSuchProcess:
                pass

    def getAllChildProcesses(self):
        parent = psutil.Process(self.process.pid)
        processes = parent.children(recursive=True)
        processes.append(parent)
        return processes

    def stopProcessBothMethods(self):
        processes = self.getAllChildProcesses()

        getLogger().info(f"Stopping {len(processes)} processes for the task.")

        # First send all the processes in the tree the terminate signal and hope they exit gracefully
        if self.process.returncode is None:
            self.gracefullyTerminateProcess(processes)
            time.sleep(3)

            # Now give any processes still alive a hard kill signal.
            self.hardKillProcess(processes)
            time.sleep(1)


    def extractResultFromOutput(self):
        if TaskProcess.resultStartString not in self.output or TaskProcess.resultFinishString not in self.output:
            getLogger().error(f"Error! Unable to extract result from the subprocess. Its possible the subprocess may have died")
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
        while self.alive:
            time.sleep(1)

        result = self.extractResultFromOutput()
        getLogger().info(f"Task Subprocess finished and gave back result:\n{json.dumps(result, indent=4)}")

        return result


    def outputMonitoringThread(self):
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

        getLogger().info(f"Terminating task subprocess {self.process.pid}, task finished.")
        self.alive = False
        self.stopProcessBothMethods()

        additionalOutput = self.getLatestLogOutput()
        if additionalOutput is not None:
            self.output += additionalOutput

        self.process.terminate()
        self.processOutputFile.close()

        getLogger().info(f"Monitoring thread has finished for {self.process.pid}")

    def timeoutMonitoringThread(self):
        while self.alive:
            elapsedSeconds = (datetime.now() - self.startTime).total_seconds()
            if elapsedSeconds > self.timeout:
                getLogger().error(f"Killing Process due to too much time elapsed")
                self.stopProcessBothMethods()

            time.sleep(1)

    def ready(self):
        return not self.alive

    def successful(self):
        return True

    def failed(self):
        return False

