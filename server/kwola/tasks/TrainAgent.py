from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from .RunTrainingStep import runTrainingStep
from .RunTestingSequence import runTestingSequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

def runRandomInitialization():
    print("Starting random testing sequences for initialization")

    # Seed the pot with 50 random sequences
    numInitializationSequences = 50
    numWorkers = 10

    futures = []
    with ProcessPoolExecutor(max_workers=numWorkers) as executor:
        for n in range(numInitializationSequences):
            sequence = TestingSequenceModel()
            sequence.save()

            future = executor.submit(runTestingSequence, str(sequence.id), True)
            futures.append(future)

        for future in as_completed(futures[:int(numInitializationSequences/2)]):
            result = future.result()
            print("Random Testing Sequence Completed")
    print("Random initialization completed")


def runMainTrainingLoop():
    sequencesNeeded = 1000
    sequencesCompleted = 0
    while sequencesCompleted < sequencesNeeded:

        sequence = TestingSequenceModel()
        sequence.save()

        print("Starting Training Step")

        runTrainingStep()

        print("Training Step Completed")

        runTestingSequence(str(sequence.id), shouldBeRandom=False)

        print("Testing Sequence Completed")

        sequencesCompleted += 1


@app.task
def trainAgent():
    # runRandomInitialization()
    runMainTrainingLoop()



if __name__ == "__main__":
    trainAgent()

