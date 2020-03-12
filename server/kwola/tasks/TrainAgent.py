from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.TrainingSequenceModel import TrainingSequence
from kwola.models.TrainingStepModel import TrainingStep
from .RunTrainingStep import runTrainingStep
from .RunTestingSequence import runTestingSequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from kwola.components.ManagedTaskSubprocess import ManagedTaskSubprocess
import time
import psutil
import subprocess
import datetime
import atexit
import bson
from kwola.config import config

def runRandomInitialization(trainingSequence):
    print("Starting random testing sequences for initialization", flush=True)

    agentConfig = config.getAgentConfiguration()

    trainingSequence.initializationTestingSequences = []

    futures = []
    with ProcessPoolExecutor(max_workers=agentConfig['training_random_initialization_workers']) as executor:
        for n in range(agentConfig['training_random_initialization_sequences']):
            sequence = TestingSequenceModel()
            sequence.save()

            trainingSequence.initializationTestingSequences.append(sequence)

            future = executor.submit(runTestingSequence, str(sequence.id), True)
            futures.append(future)

            # Add in a delay for each successive task so that they parallelize smoother
            # without fighting for CPU during the startup of that task
            time.sleep(3)

        for future in as_completed(futures[:int(agentConfig['training_random_initialization_sequences']/2)]):
            result = future.result()
            print("Random Testing Sequence Completed", flush=True)

    # Save the training sequence with all the data on the initialization sequences
    trainingSequence.save()
    print("Random initialization completed", flush=True)




def runTrainingSubprocess(trainingSequence):
    process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTrainingStep"], {
        "trainingSequenceId": str(trainingSequence.id)
    }, timeout=config.getAgentConfiguration()['training_step_timeout'])

    result = process.waitForProcessResult()


    trainingStepId = str(result['trainingStepId'])
    trainingStep = TrainingStep.objects({id: bson.ObjectId(trainingStepId)}).first()
    trainingSequence.trainingSteps.append(trainingStep)
    trainingSequence.save()

    return


def runTestingSubprocess(trainingSequence):
    testingSequence = TestingSequenceModel()
    testingSequence.save()


    process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTestingSequence"], {
        "testingSequenceId": str(testingSequence.id),
        "shouldBeRandom": False
    }, timeout=config.getAgentConfiguration()['training_step_timeout'])
    result = process.waitForProcessResult()


    # Reload the testing sequence from the db. It will have been updated by the sub-process.
    testingSequence = TestingSequenceModel.objects({id: bson.ObjectId(testingSequence.id)}).first()
    trainingSequence.testingSequences.append(testingSequence)
    trainingSequence.save()

    return



def runMainTrainingLoop(trainingSequence):
    agentConfig = config.getAgentConfiguration()

    stepsCompleted = 0

    stepStartTime = datetime.datetime.now()

    while stepsCompleted < agentConfig['total_training_steps_needed']:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            trainingFuture = executor.submit(runTrainingSubprocess)
            futures.append(trainingFuture)

            for testingSequences in range(agentConfig['testing_sequences_per_training_step']):
                futures.append(executor.submit(runTestingSubprocess))

            wait(futures)

            print("Completed one parallel training & testing step! Hooray!", flush=True)

            stepsCompleted += 1

        trainingSequence.trainingStepsCompleted += 1
        trainingSequence.averageTimePerStep = (datetime.datetime.now() - stepStartTime).total_seconds() / stepsCompleted
        trainingSequence.save()


@app.task
def trainAgent():
    trainingSequence = TrainingSequence()

    trainingSequence.startTime = datetime.datetime.now()
    trainingSequence.status = "running"
    trainingSequence.trainingStepsCompleted = 0

    runRandomInitialization(trainingSequence)
    runMainTrainingLoop(trainingSequence)

    trainingSequence.status = "completed"
    trainingSequence.endTime = datetime.datetime.now()
    trainingSequence.save()


if __name__ == "__main__":
    trainAgent()

