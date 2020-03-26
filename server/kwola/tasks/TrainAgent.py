from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.TrainingSequenceModel import TrainingSequence
from kwola.models.TrainingStepModel import TrainingStep
from .RunTrainingStep import runTrainingStep
from .RunTestingSequence import runTestingSequence
from concurrent.futures import ThreadPoolExecutor
import mongoengine
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from kwola.components.ManagedTaskSubprocess import ManagedTaskSubprocess
import time
import multiprocessing
import psutil
import subprocess
import traceback
from datetime import datetime
import atexit
import bson
from kwola.config import config

def runRandomInitializationSubprocess(trainingSequence):
    testingSequence = TestingSequenceModel()
    testingSequence.save()

    process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTestingSequence"], {
        "testingSequenceId": str(testingSequence.id),
        "shouldBeRandom": True
    }, timeout=config.getAgentConfiguration()['random_initialization_testing_sequence_timeout'])
    result = process.waitForProcessResult()

    # Reload the testing sequence from the db. It will have been updated by the sub-process.
    testingSequence = TestingSequenceModel.objects(id=bson.ObjectId(testingSequence.id) ).first()
    trainingSequence.initializationTestingSequences.append(testingSequence)
    trainingSequence.save()

    return


def runRandomInitialization(trainingSequence):
    print(datetime.now(), "Starting random testing sequences for initialization", flush=True)

    agentConfig = config.getAgentConfiguration()

    trainingSequence.initializationTestingSequences = []

    futures = []
    with ThreadPoolExecutor(max_workers=agentConfig['training_random_initialization_workers']) as executor:
        for n in range(agentConfig['training_random_initialization_sequences']):
            sequence = TestingSequenceModel()
            sequence.save()

            future = executor.submit(runRandomInitializationSubprocess, trainingSequence)
            futures.append(future)

            # Add in a delay for each successive task so that they parallelize smoother
            # without fighting for CPU during the startup of that task
            time.sleep(3)

        for future in as_completed(futures):
            result = future.result()
            print(datetime.now(), "Random Testing Sequence Completed", flush=True)

    # Save the training sequence with all the data on the initialization sequences
    trainingSequence.save()
    print(datetime.now(), "Random initialization completed", flush=True)




def runTrainingSubprocess(trainingSequence, gpuNumber):
    try:
        process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTrainingStep"], {
            "trainingSequenceId": str(trainingSequence.id),
            "gpu": gpuNumber
        }, timeout=config.getAgentConfiguration()['training_step_timeout'])

        result = process.waitForProcessResult()

        if 'trainingStepId' in result:
            trainingStepId = str(result['trainingStepId'])
            trainingStep = TrainingStep.objects(id=bson.ObjectId(trainingStepId)).first()
            trainingSequence.trainingSteps.append(trainingStep)
            trainingSequence.save()
    except Exception as e:
        traceback.print_exc()
        print(datetime.now(), "Training task subprocess appears to have failed", flush=True)


def runTestingSubprocess(trainingSequence, generateDebugVideo=False):
    testingSequence = TestingSequenceModel()
    testingSequence.save()

    process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTestingSequence"], {
        "testingSequenceId": str(testingSequence.id),
        "shouldBeRandom": False,
        "generateDebugVideo": generateDebugVideo
    }, timeout=config.getAgentConfiguration()['training_step_timeout'])
    result = process.waitForProcessResult()

    # Reload the testing sequence from the db. It will have been updated by the sub-process.
    testingSequence = TestingSequenceModel.objects(id=bson.ObjectId(testingSequence.id)).first()
    trainingSequence.testingSequences.append(testingSequence)
    trainingSequence.save()



def runMainTrainingLoop(trainingSequence):
    agentConfig = config.getAgentConfiguration()

    stepsCompleted = 0

    stepStartTime = datetime.now()

    while stepsCompleted < agentConfig['training_steps_needed']:
        with ThreadPoolExecutor(max_workers=(agentConfig['testing_sequences_in_parallel_per_training_step'] + 2)) as executor:
            futures = []

            trainingFuture = executor.submit(runTrainingSubprocess, trainingSequence, gpuNumber=0)
            futures.append(trainingFuture)

            trainingFuture = executor.submit(runTrainingSubprocess, trainingSequence, gpuNumber=1)
            futures.append(trainingFuture)

            for testingSequenceNumber in range(agentConfig['testing_sequences_per_training_step']):
                futures.append(executor.submit(runTestingSubprocess, trainingSequence, generateDebugVideo=True if testingSequenceNumber == 0 else False))
                time.sleep(3)

            wait(futures)

            print(datetime.now(), "Completed one parallel training & testing step! Hooray!", flush=True)

            stepsCompleted += 1

        trainingSequence.trainingStepsCompleted += 1
        trainingSequence.averageTimePerStep = (datetime.now() - stepStartTime).total_seconds() / stepsCompleted
        trainingSequence.save()


def trainAgent():
    multiprocessing.set_start_method('spawn')

    trainingSequence = TrainingSequence()

    trainingSequence.startTime = datetime.now()
    trainingSequence.status = "running"
    trainingSequence.trainingStepsCompleted = 0

    runRandomInitialization(trainingSequence)
    runMainTrainingLoop(trainingSequence)

    trainingSequence.status = "completed"
    trainingSequence.endTime = datetime.now()
    trainingSequence.save()


@app.task
def trainAgentTask():
    trainAgent()


if __name__ == "__main__":
    mongoengine.connect('kwola')
    trainAgent()

