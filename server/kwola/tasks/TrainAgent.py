from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from .RunTrainingStep import runTrainingStep
from .RunTestingSequence import runTestingSequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time



@app.task
def trainAgent():
    print("Starting random testing sequences for initialization")

    # Seed the pot with 10 random sequences
    # initializationSequences = 10
    initializationSequences = 1
    futures = []
    # with ProcessPoolExecutor(max_workers=10) as executor:

    for n in range(initializationSequences):
        sequence = TestingSequenceModel()
        sequence.save()

        # future = executor.submit(runTestingSequence, str(sequence.id), True)
        # futures.append(future)


        runTestingSequence(str(sequence.id), True)

        # We leave a gap in the start time between each process to provide time for startup
        # time.sleep(30)

    # for future in futures:
    #     result = future.result()
    #     print("Random Testing Sequence Completed")

    print("Random initialization completed")

    sequencesNeeded = 1000
    sequencesCompleted = 0
    while sequencesCompleted < sequencesNeeded:

        sequence = TestingSequenceModel()
        sequence.save()

        runTrainingStep()

        print("Training Step Completed")

        runTestingSequence(str(sequence.id), shouldBeRandom=False)

        print("Testing Sequence Completed")

        sequencesCompleted += 1



if __name__ == "__main__":
    trainAgent()

