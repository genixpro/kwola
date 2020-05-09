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
from ..datamodels.ExecutionSessionModel import ExecutionSession
from ..datamodels.ExecutionTraceModel import ExecutionTrace
from .RunTrainingStep import loadAllTestingSteps
from concurrent.futures import as_completed, wait
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import billiard as multiprocessing
import os
import os.path
import time
import torch.cuda
import traceback
import random


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

    return result


def runRandomInitialization(config, trainingSequence, exitOnFail=True):
    print(datetime.now(), f"[{os.getpid()}]", "Starting random testing sequences for initialization", flush=True)

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
            if result is None or ('success' in result and not result['success']):
                if exitOnFail:
                    raise RuntimeError("Random initialization sequence failed and did not return a result.")
            else:
                updateModelSymbols(config, result['testingStepId'])

            print(datetime.now(), f"[{os.getpid()}]", "Random Testing Sequence Completed", flush=True)

    # Save the training sequence with all the data on the initialization sequences
    trainingSequence.saveToDisk(config)
    print(datetime.now(), f"[{os.getpid()}]", "Random initialization completed", flush=True)


def runTrainingSubprocess(config, trainingSequence, trainingStepIndex, gpuNumber, coordinatorTempFileName):
    try:
        process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTrainingStep"], {
            "configDir": config.configurationDirectory,
            "trainingSequenceId": str(trainingSequence.id),
            "trainingStepIndex": trainingStepIndex,
            "gpu": gpuNumber,
            "coordinatorTempFileName": coordinatorTempFileName
        }, timeout=config['training_step_timeout'], config=config, logId=str(trainingSequence.id + "_training_step_" + str(trainingStepIndex)))

        result = process.waitForProcessResult()

        if result is not None and 'trainingStepId' in result:
            trainingStepId = str(result['trainingStepId'])
            trainingStep = TrainingStep.loadFromDisk(trainingStepId, config)
            trainingSequence.trainingSteps.append(trainingStep)
            trainingSequence.saveToDisk(config)
        else:
            print(datetime.now(), f"[{os.getpid()}]", "Training task subprocess appears to have failed", flush=True)

        result['finishTime'] = datetime.now()

        return result

    except Exception as e:
        traceback.print_exc()
        print(datetime.now(), f"[{os.getpid()}]", "Training task subprocess appears to have failed", flush=True)


def runTestingSubprocess(config, trainingSequence, testStepIndex, generateDebugVideo=False):
    testingStep = TestingStep(id=str(trainingSequence.id + "_testing_step_" + str(testStepIndex)))
    testingStep.saveToDisk(config)

    process = ManagedTaskSubprocess(["python3", "-m", "kwola.tasks.RunTestingStep"], {
        "configDir": config.configurationDirectory,
        "testingStepId": str(testingStep.id),
        "shouldBeRandom": False,
        "generateDebugVideo": generateDebugVideo and config['enable_debug_videos']
    }, timeout=config['training_step_timeout'], config=config, logId=testingStep.id)
    result = process.waitForProcessResult()

    # Reload the testing sequence from the db. It will have been updated by the sub-process.
    testingStep = TestingStep.loadFromDisk(testingStep.id, config)
    trainingSequence.testingSteps.append(testingStep)
    trainingSequence.saveToDisk(config)

    result['finishTime'] = datetime.now()

    return result

def updateModelSymbols(config, testingStepId):
    # Load and save the agent to make sure all training subprocesses are synced
    agent = DeepLearningAgent(config=config, whichGpu=None)
    agent.initialize(enableTraining=False)
    agent.load()

    testingStep = TestingStep.loadFromDisk(testingStepId, config)

    totalNewSymbols = 0
    for executionSessionId in testingStep.executionSessions:
        executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

        traces = []
        for executionTraceId in executionSession.executionTraces:
            traces.append(ExecutionTrace.loadFromDisk(executionTraceId, config))

        totalNewSymbols += agent.assignNewSymbols(traces)

    print(datetime.now(), f"[{os.getpid()}]", f"Added {totalNewSymbols} new symbols from testing step {testingStepId}", flush=True)

    agent.save()

def runMainTrainingLoop(config, trainingSequence, exitOnFail=False):
    stepsCompleted = 0

    stepStartTime = datetime.now()

    testStepsLaunched = 0
    trainingStepsLaunched = 0

    numberOfTrainingStepsInParallel = max(1, torch.cuda.device_count())

    while stepsCompleted < config['training_steps_needed']:
        with ThreadPoolExecutor(max_workers=(config['testing_sequences_in_parallel_per_training_step'] + numberOfTrainingStepsInParallel)) as executor:
            coordinatorTempFileName = "kwola_distributed_coordinator-" + str(random.randint(0, 1e8))
            coordinatorTempFilePath = "/tmp/" + coordinatorTempFileName
            if os.path.exists(coordinatorTempFilePath):
                os.unlink(coordinatorTempFilePath)

            allFutures = []
            testStepFutures = []
            trainStepFutures = []

            if torch.cuda.device_count() > 0:
                for gpu in range(numberOfTrainingStepsInParallel):
                    trainingFuture = executor.submit(runTrainingSubprocess, config, trainingSequence, trainingStepIndex=trainingStepsLaunched, gpuNumber=gpu, coordinatorTempFileName=coordinatorTempFileName)
                    allFutures.append(trainingFuture)
                    trainStepFutures.append(trainingFuture)
                    trainingStepsLaunched += 1
            else:
                trainingFuture = executor.submit(runTrainingSubprocess, config, trainingSequence, trainingStepIndex=trainingStepsLaunched, gpuNumber=None, coordinatorTempFileName=coordinatorTempFileName)
                allFutures.append(trainingFuture)
                trainStepFutures.append(trainingFuture)
                trainingStepsLaunched += 1

            for testingStepNumber in range(config['testing_sequences_per_training_step']):
                testStepIndex = testStepsLaunched + config['training_random_initialization_sequences']
                future = executor.submit(runTestingSubprocess, config, trainingSequence, testStepIndex, generateDebugVideo=True if testingStepNumber == 0 else False)
                allFutures.append(future)
                testStepFutures.append(future)
                time.sleep(3)
                testStepsLaunched += 1

            wait(allFutures)

            anyFailures = False
            for future in allFutures:
                result = future.result()
                if result is None:
                    anyFailures = True
                if 'success' in result and not result['success']:
                    anyFailures = True

            if anyFailures and exitOnFail:
                raise RuntimeError("One of the testing / training loops failed and did not return successfully. Exiting the training loop.")

            if not anyFailures:
                print(datetime.now(), f"[{os.getpid()}]", "Updating the symbols table", flush=True)
                for future in testStepFutures:
                    result = future.result()
                    testingStepId = result['testingStepId']
                    updateModelSymbols(config, testingStepId)

                # Here, we dynamically adjust the number of iterations to be executed in a training step
                # so that it aligns automatically with the time needed to complete the testing steps
                lastTestFinish = None
                lastTrainFinish = None
                for future in testStepFutures:
                    if lastTestFinish is None or lastTestFinish < future.result()['finishTime']:
                        lastTestFinish = future.result()['finishTime']

                for future in trainStepFutures:
                    if lastTrainFinish is None or lastTrainFinish < future.result()['finishTime']:
                        lastTrainFinish = future.result()['finishTime']

                if lastTestFinish > lastTrainFinish:
                    config['iterations_per_training_step'] += config['iterations_per_training_step_adjustment_size_per_loop']
                else:
                    config['iterations_per_training_step'] = max(5, config['iterations_per_training_step'] - config['iterations_per_training_step_adjustment_size_per_loop'])
                config.saveConfig()

            print(datetime.now(), f"[{os.getpid()}]", "Completed one parallel training & testing step! Hooray!", flush=True)

            stepsCompleted += 1

            time.sleep(3)

        trainingSequence.trainingStepsCompleted += 1
        trainingSequence.averageTimePerStep = (datetime.now() - stepStartTime).total_seconds() / stepsCompleted
        trainingSequence.saveToDisk(config)


def trainAgent(configDir, exitOnFail=False):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    config = Configuration(configDir)

    # Create the bugs directory. This is just temporary
    config.getKwolaUserDataDirectory("bugs")

    # Load and save the agent to make sure all training subprocesses are synced
    agent = DeepLearningAgent(config=config, whichGpu=None)
    agent.initialize(enableTraining=False)
    agent.load()
    agent.save()
    del agent

    # Create and destroy an environment, which forces a lot of the initial javascript in the application
    # to be loaded and translated. It also just verifies that the system can access the target URL prior
    # to trying to run a full sequence
    environment = WebEnvironment(config, sessionLimit=1)
    environment.shutdown()
    del environment

    trainingSequence = TrainingSequence(id=CustomIDField.generateNewUUID(TrainingSequence, config))

    trainingSequence.startTime = datetime.now()
    trainingSequence.status = "running"
    trainingSequence.trainingStepsCompleted = 0
    trainingSequence.saveToDisk(config)

    testingSteps = [step for step in loadAllTestingSteps(config) if step.status == "completed"]
    if len(testingSteps) == 0:
        runRandomInitialization(config, trainingSequence, exitOnFail=exitOnFail)
        trainingSequence.saveToDisk(config)

    runMainTrainingLoop(config, trainingSequence, exitOnFail=exitOnFail)

    trainingSequence.status = "completed"
    trainingSequence.endTime = datetime.now()
    trainingSequence.saveToDisk(config)
