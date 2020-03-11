from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from .RunTrainingStep import runTrainingStep
from .RunTestingSequence import runTestingSequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
import time
import subprocess

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


def runTrainingSubprocess():
    subprocess.run(["python3", "-m", "kwola.tasks.RunTrainingStep"])

def runTestingSubprocess():
    subprocess.run(["python3", "-m", "kwola.tasks.RunTestingSequence"])

def runMainTrainingLoop():
    sequencesNeeded = 1000
    sequencesCompleted = 0
    numTestSequencesPerTrainingStep = 1
    while sequencesCompleted < sequencesNeeded:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            trainingFuture = executor.submit(runTrainingSubprocess)
            futures.append(trainingFuture)

            for testingSequences in range(numTestSequencesPerTrainingStep):
                futures.append(executor.submit(runTestingSubprocess))

            wait(futures)

            print("Completed one parallel training loop! Hooray!")

            sequencesCompleted += 1


@app.task
def trainAgent():
    # runRandomInitialization()
    runMainTrainingLoop()



if __name__ == "__main__":
    trainAgent()

