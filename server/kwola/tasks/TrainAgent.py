from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from .RunTrainingStep import runTrainingStep
from .RunTestingSequence import runTestingSequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import time
import psutil
import subprocess
import datetime
import atexit

def runRandomInitialization():
    print("Starting random testing sequences for initialization")

    # Seed the pot with 10 random sequences
    numInitializationSequences = 10
    numWorkers = 2

    futures = []
    with ProcessPoolExecutor(max_workers=numWorkers) as executor:
        for n in range(numInitializationSequences):
            sequence = TestingSequenceModel()
            sequence.save()

            future = executor.submit(runTestingSequence, str(sequence.id), True)
            futures.append(future)

            # Add in a delay for each successive task so that they parallelize smoother
            # without fighting for CPU during the startup of that task
            time.sleep(3)

        for future in as_completed(futures[:int(numInitializationSequences/2)]):
            result = future.result()
            print("Random Testing Sequence Completed")

    print("Random initialization completed")

def recursiveKillProcess(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        children.append(parent)
        for p in children:
            p.send_signal(9)
    except psutil.NoSuchProcess:
        pass

def waitOnProcess(process, finishText, timeLimit=800):
    atexit.register(lambda: process.kill())

    output = ''
    startTime = datetime.datetime.now()
    while process.returncode is None and (finishText not in output):
        nextChars = str(process.stdout.readline(), 'utf8')
        elapsedSeconds = (datetime.datetime.now() - startTime).total_seconds()
        for nextChar in nextChars:
            if nextChar == chr(127):
                output = output[:-1]  # Erase the last character from the output.
            else:
                output += nextChar
                print(nextChar, sep="", end="")

        print("", sep="", end="", flush=True)

        if elapsedSeconds > timeLimit:
            print("Killing Process due to too much time elapsed")
            recursiveKillProcess(process)
            break
        
        time.sleep(0.002)

    time.sleep(1)
    print("Terminating process, task finished.")

    # DESTROY IT!
    if process.returncode is None:
        process.terminate()
        time.sleep(3)

    if process.returncode is None:
        recursiveKillProcess(process)


def runTrainingSubprocess():
    process = subprocess.Popen(["python3", "-m", "kwola.tasks.RunTrainingStep"], stdout=subprocess.PIPE, stderr=None, stdin=subprocess.PIPE)

    waitOnProcess(process, "==== Training Step Completed ====")

def runTestingSubprocess():
    process = subprocess.Popen(["python3", "-m", "kwola.tasks.RunTestingSequence"], stdout=subprocess.PIPE, stderr=None, stdin=subprocess.PIPE)

    waitOnProcess(process, "==== Finished Running Testing Sequence! ====")

def runMainTrainingLoop():
    loopsNeeded = 1000
    loopsCompleted = 0
    numTestSequencesPerLoop = 1
    while loopsCompleted < loopsNeeded:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            trainingFuture = executor.submit(runTrainingSubprocess)
            futures.append(trainingFuture)

            for testingSequences in range(numTestSequencesPerLoop):
                futures.append(executor.submit(runTestingSubprocess))

            wait(futures)

            print("Completed one parallel training loop! Hooray!")

            loopsCompleted += 1


@app.task
def trainAgent():
    # runRandomInitialization()
    runMainTrainingLoop()



if __name__ == "__main__":
    trainAgent()

