import subprocess
import psutil
import atexit
import time
import json
import threading
import datetime
from kwola.components.TaskProcess import TaskProcess


class ManagedTaskSubprocess:
    """
        This class is used to manage Kwola task subprocesses. These are subprocesses used in Kwola to due specific
        resource heavy tasks.

        They communicate with the master "manager" (this class) using a very simple JSON line-by-line communication
        scheme using just regular standard input / output.

        They will also be monitored with a timeout and such.
    """

    def __init__(self, args, data, timeout):
        self.process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=None, stdin=subprocess.PIPE)

        self.process.stdin.write(bytes(json.dumps(data) + "\n", "utf8"))
        self.process.stdin.flush()
        self.startTime = datetime.datetime.now()
        self.alive = True
        self.timeout = timeout

        self.output = ''

        self.monitorTimeoutProcess = threading.Thread(target=lambda: self.timeoutMonitoringThread())


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
        resultStart = self.output.index(TaskProcess.resultStartString)
        resultFinish = self.output.index(TaskProcess.resultFinishString)

        if resultStart is None or resultFinish is None:
            print("Error! Unable to exact result from the subprocess. Possible it may have died", flush=True)
            return None
        else:
            resultDataString = self.output[resultStart + len(TaskProcess.resultStartString) : resultFinish]
            result = json.loads(resultDataString)
            return result


    def waitForProcessResult(self):
        atexit.register(lambda: self.process.kill())

        self.output = ''
        while self.process.returncode is None and (TaskProcess.resultFinishString not in self.output) and self.alive:
            nextChars = str(self.process.stdout.readline(), 'utf8')
            for nextChar in nextChars:
                if nextChar == chr(127):
                    self.output = self.output[:-1]  # Erase the last character from the self.output.
                else:
                    self.output += nextChar
                    print(nextChar, sep="", end="")

            print("", sep="", end="", flush=True)

        print("Terminating process, task finished.", flush=True)
        self.alive = False
        self.stopProcessBothMethods()

        additionalOutput = str(self.process.stdout.read(), 'utf8')
        self.output += additionalOutput
        print(additionalOutput, sep="", end="", flush=True)

        result = self.extractResultFromOutput()
        print("Task Subprocess finished and gave back result", flush=True)
        print(json.dumps(result, indent=4), flush=True)

        return result



    def timeoutMonitoringThread(self):
        while self.alive:
            elapsedSeconds = (datetime.datetime.now() - self.startTime).total_seconds()
            if elapsedSeconds > self.timeout:
                print("Killing Process due to too much time elapsed", flush=True)
                self.stopProcessBothMethods()

            time.sleep(1)

