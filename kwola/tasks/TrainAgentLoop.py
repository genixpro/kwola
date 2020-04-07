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


from ..components.agents.DeepLearningAgent import DeepLearningAgent
from ..components.environments.WebEnvironment import WebEnvironment
from ..tasks.ManagedTaskSubprocess import ManagedTaskSubprocess
from ..config.config import Configuration
from ..datamodels.CustomIDField import CustomIDField
from ..datamodels.TestingStepModel import TestingStep
from ..datamodels.TrainingSequenceModel import TrainingSequence
from ..datamodels.TrainingStepModel import TrainingStep
from concurrent.futures import as_completed, wait
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import multiprocessing
import os
import os.path
import time
import torch.cuda
import traceback


def runRandomInitializationSubprocess(config, trainingSequence, testStepIndex):
    testingStep = TestingStep(id=str(trainingSequence.id + "_testing_step_" + str(testStepIndex)))
    testingStep.saveToDisk(config)

    process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTestingStep"], {
        "configDir": config.configurationDirectory,
        "testingStepId": str(testingStep.id),
        "shouldBeRandom": True
    }, timeout=config['random_initialization_testing_sequence_timeout'], config=config, logId=testingStep.id)
    result = process.waitForProcessResult()

    # Reload the testing sequence from the db. It will have been updated by the sub-process.
    testingStep = TestingStep.loadFromDisk(testingStep.id, config)
    trainingSequence.initializationTestingSteps.append(testingStep)
    trainingSequence.saveToDisk(config)

    return


def runRandomInitialization(config, trainingSequence):
    print(datetime.now(), "Starting random testing sequences for initialization", flush=True)

    trainingSequence.initializationTestingSteps = []

    futures = []
    with ThreadPoolExecutor(max_workers=config['training_random_initialization_workers']) as executor:
        for testStepIndex in range(config['training_random_initialization_sequences']):
            future = executor.submit(runRandomInitializationSubprocess, config, trainingSequence, testStepIndex)
            futures.append(future)

            # Add in a delay for each successive task so that they parallelize smoother
            # without fighting for CPU during the startup of that task
            time.sleep(3)

        for future in as_completed(futures):
            result = future.result()
            print(datetime.now(), "Random Testing Sequence Completed", flush=True)

    # Save the training sequence with all the data on the initialization sequences
    trainingSequence.saveToDisk(config)
    print(datetime.now(), "Random initialization completed", flush=True)


def runTrainingSubprocess(config, trainingSequence, trainingStepIndex, gpuNumber):
    try:
        process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTrainingStep"], {
            "configDir": config.configurationDirectory,
            "trainingSequenceId": str(trainingSequence.id),
            "trainingStepIndex": trainingStepIndex,
            "gpu": gpuNumber
        }, timeout=config['training_step_timeout'], config=config, logId=str(trainingSequence.id + "_training_step_" + str(trainingStepIndex)))

        result = process.waitForProcessResult()

        if result is not None and 'trainingStepId' in result:
            trainingStepId = str(result['trainingStepId'])
            trainingStep = TrainingStep.loadFromDisk(trainingStepId, config)
            trainingSequence.trainingSteps.append(trainingStep)
            trainingSequence.saveToDisk(config)
        else:
            print(datetime.now(), "Training task subprocess appears to have failed", flush=True)

    except Exception as e:
        traceback.print_exc()
        print(datetime.now(), "Training task subprocess appears to have failed", flush=True)


def runTestingSubprocess(config, trainingSequence, testStepIndex, generateDebugVideo=False):
    testingStep = TestingStep(id=str(trainingSequence.id + "_testing_step_" + str(testStepIndex)))
    testingStep.saveToDisk(config)

    process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTestingStep"], {
        "configDir": config.configurationDirectory,
        "testingStepId": str(testingStep.id),
        "shouldBeRandom": False,
        "generateDebugVideo": generateDebugVideo
    }, timeout=config['training_step_timeout'], config=config, logId=testingStep.id)
    result = process.waitForProcessResult()

    # Reload the testing sequence from the db. It will have been updated by the sub-process.
    testingStep = TestingStep.loadFromDisk(testingStep.id, config)
    trainingSequence.testingSteps.append(testingStep)
    trainingSequence.saveToDisk(config)


def runMainTrainingLoop(config, trainingSequence):
    # Load and save the agent to make sure all training subprocesses are synced
    environment = WebEnvironment(config=config, sessionLimit=1)
    agent = DeepLearningAgent(config=config, whichGpu=None)
    agent.initialize(environment.branchFeatureSize())
    agent.load()
    agent.save()
    del environment, agent

    stepsCompleted = 0

    stepStartTime = datetime.now()

    testStepsLaunched = 0
    trainingStepsLaunched = 0

    numberOfTrainingStepsInParallel = max(1, torch.cuda.device_count())

    while stepsCompleted < config['training_steps_needed']:
        with ThreadPoolExecutor(max_workers=(config['testing_sequences_in_parallel_per_training_step'] + numberOfTrainingStepsInParallel)) as executor:
            if os.path.exists("/tmp/kwola_distributed_coordinator"):
                os.unlink("/tmp/kwola_distributed_coordinator")

            futures = []

            if torch.cuda.device_count() > 0:
                for gpu in range(numberOfTrainingStepsInParallel):
                    trainingFuture = executor.submit(runTrainingSubprocess, config, trainingSequence, trainingStepIndex=trainingStepsLaunched, gpuNumber=gpu)
                    futures.append(trainingFuture)
                    trainingStepsLaunched += 1
            else:
                trainingFuture = executor.submit(runTrainingSubprocess, config, trainingSequence, trainingStepIndex=trainingStepsLaunched, gpuNumber=None)
                futures.append(trainingFuture)
                trainingStepsLaunched += 1

            for testingStepNumber in range(config['testing_sequences_per_training_step']):
                testStepIndex = testStepsLaunched + config['training_random_initialization_sequences']
                futures.append(executor.submit(runTestingSubprocess, config, trainingSequence, testStepIndex, generateDebugVideo=True if testingStepNumber == 0 else False))
                time.sleep(3)
                testStepsLaunched += 1

            wait(futures)

            print(datetime.now(), "Completed one parallel training & testing step! Hooray!", flush=True)

            stepsCompleted += 1

            time.sleep(3)

        trainingSequence.trainingStepsCompleted += 1
        trainingSequence.averageTimePerStep = (datetime.now() - stepStartTime).total_seconds() / stepsCompleted
        trainingSequence.saveToDisk(config)


def trainAgent(configDir):
    multiprocessing.set_start_method('spawn')

    config = Configuration(configDir)

    trainingSequence = TrainingSequence(id=CustomIDField.generateNewUUID(TrainingSequence, config))

    trainingSequence.startTime = datetime.now()
    trainingSequence.status = "running"
    trainingSequence.trainingStepsCompleted = 0

    runRandomInitialization(config, trainingSequence)
    runMainTrainingLoop(config, trainingSequence)

    trainingSequence.status = "completed"
    trainingSequence.endTime = datetime.now()
    trainingSequence.saveToDisk(config)
