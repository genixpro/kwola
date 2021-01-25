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
from ..components.agents.DeepLearningAgent import DeepLearningAgent
from ..components.agents.SymbolMapper import SymbolMapper
from ..components.environments.WebEnvironment import WebEnvironment
from ..tasks.ManagedTaskSubprocess import ManagedTaskSubprocess
from ..config.config import KwolaCoreConfiguration
from ..datamodels.CustomIDField import CustomIDField
from ..datamodels.TestingStepModel import TestingStep
from ..datamodels.TrainingSequenceModel import TrainingSequence
from ..datamodels.TrainingStepModel import TrainingStep
from ..datamodels.ExecutionSessionModel import ExecutionSession
from ..datamodels.ExecutionTraceModel import ExecutionTrace
from ..components.managers.TrainingManager import TrainingManager
from ..components.utils.charts import generateAllCharts
from ..components.utils.asyncthreadfuture import AsyncThreadFuture
from concurrent.futures import as_completed, wait
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import billiard as multiprocessing
import os
import os.path
import time
import shutil
import torch.cuda
import traceback
import random
import subprocess
import sys
import tempfile

def getAvailableBrowsers(config):
    browsers = []
    if config['web_session_enable_chrome']:
        try:
            result = subprocess.run(['chromedriver', '-v'], stdout=subprocess.PIPE)
        except FileNotFoundError:
            result = None

        try:
            result2 = subprocess.run(['chromium-browser', '--version'], stdout=subprocess.PIPE)
        except FileNotFoundError:
            result2 = None

        try:
            chromeCmd = "google-chrome"
            if sys.platform == "win32" or sys.platform == "win64":
                chromeCmd = "C:\Program Files\Google\Chrome\Application\chrome.exe"

                if not os.path.exists(chromeCmd):
                    chromeCmd = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            result3 = subprocess.run([chromeCmd, '--headless', '--version'], stdout=subprocess.PIPE)
        except FileNotFoundError:
            result3 = None

        if result is not None and (result2 is not None or result3 is not None):
            browsers.append("chrome")
        else:
            getLogger().error(f"The Chrome browser is enabled in the configuration, but the executables for either chromedriver or google-chrome/chromium-browser can not be found in $PATH. PATH is:\n{os.getenv('PATH')}")

    if config['web_session_enable_firefox']:
        try:
            result = subprocess.run(['geckodriver', '--version'], stdout=subprocess.PIPE)
        except FileNotFoundError:
            result = None

        try:
            firefoxCmd = "firefox"
            if sys.platform == "win32" or sys.platform == "win64":
                firefoxCmd = "C:\Program Files\Mozilla Firefox\firefox.exe"
            result2 = subprocess.run([firefoxCmd, '--version'], stdout=subprocess.PIPE)
        except FileNotFoundError:
            result2 = None

        if result is not None and result2 is not None:
            browsers.append("firefox")
        else:
            getLogger().error(f"The Firefox browser is enabled in the configuration, but the executables for either geckodriver or firefox can not be found in $PATH. PATH is:\n{os.getenv('PATH')}")

    if config['web_session_enable_edge']:
        try:
            result = subprocess.run(['msedgedriver', '--version'], stdout=subprocess.PIPE)
        except FileNotFoundError:
            result = None

        try:
            edgeCmd = "microsoft-edge"
            if sys.platform == "win32" or sys.platform == "win64":
                edgeCmd = "C:\Program Files (x86)\Microsoft\Edge\Application\edge.exe"

            result2 = subprocess.run([edgeCmd, '--version'], stdout=subprocess.PIPE)
        except FileNotFoundError:
            result2 = None

        if result is not None and result2 is not None:
            browsers.append("edge")
        else:
            getLogger().error(f"The Microsoft Edge browser is enabled in the configuration, but the executables for either msedgedriver or microsoft-edge can not be found in $PATH. PATH is:\n{os.getenv('PATH')}")

    return browsers


def getAvailableWindowSizes(config):
    windowSizes = []

    if config['web_session_enable_window_size_desktop']:
        windowSizes.append("desktop")
    if config['web_session_enable_window_size_tablet']:
        windowSizes.append("tablet")
    if config['web_session_enable_window_size_mobile']:
        windowSizes.append("mobile")

    if len(windowSizes) == 0:
        raise RuntimeError("Error! There are no enabled window sizes. Please set web_session_enable_window_size_desktop or web_session_enable_window_size_tablet or web_session_enable_window_size_mobile")

    return windowSizes

def runRandomInitializationSubprocess(config, trainingSequence, testStepIndex):
    try:
        browsers = getAvailableBrowsers(config)
        windowSizes = getAvailableWindowSizes(config)

        choiceIndex = testStepIndex % (len(browsers) * len(windowSizes))
        chosenBrowser = browsers[int(choiceIndex / len(windowSizes))]
        chosenWindowSize = windowSizes[choiceIndex % len(windowSizes)]

        testingStep = TestingStep(id=str(trainingSequence.id + "_testing_step_" + str(testStepIndex)), browser=chosenBrowser, windowSize=chosenWindowSize)
        testingStep.saveToDisk(config)

        process = ManagedTaskSubprocess([sys.executable, "-m", "kwola.tasks.RunTestingStep"], {
            "config": config.serialize(),
            "testingStepId": str(testingStep.id),
            "shouldBeRandom": True,
            "generateDebugVideo": False,
            "browser": chosenBrowser,
            "windowSize": chosenWindowSize
        }, timeout=config['random_initialization_testing_sequence_timeout'], config=config, logId=testingStep.id)
        process.start()
        result = process.waitForProcessResult()

        # Reload the testing sequence from the db. It will have been updated by the sub-process.
        trainingSequence.initializationTestingSteps.append(testingStep.id)
        trainingSequence.saveToDisk(config)

        return result
    except Exception as e:
        getLogger().error(f"Testing task subprocess appears to have failed. {traceback.format_exc()}")
        raise


def runRandomInitialization(config, trainingSequence, exitOnFail=True):
    getLogger().info(f"Starting random testing sequences for initialization")

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

            getLogger().info(f"Random Testing Sequence Completed")

    getLogger().info(f"Random initialization completed")


def runTrainingSubprocess(config, trainingSequence, trainingStepIndex, gpuNumber, coordinatorTempFileName):
    try:
        process = ManagedTaskSubprocess([sys.executable, "-m", "kwola.tasks.RunTrainingStep"], {
            "config": config.serialize(),
            "trainingSequenceId": str(trainingSequence.id),
            "trainingStepIndex": trainingStepIndex,
            "gpu": gpuNumber,
            "coordinatorTempFileName": coordinatorTempFileName
        }, timeout=config['training_step_timeout'], config=config, logId=str(trainingSequence.id + "_training_step_" + str(trainingStepIndex)))

        process.start()

        result = process.waitForProcessResult()

        if result is not None and 'trainingStepId' in result:
            result['finishTime'] = datetime.now()

            trainingStepId = str(result['trainingStepId'])
            trainingSequence.trainingSteps.append(trainingStepId)
            trainingSequence.saveToDisk(config)
        else:
            getLogger().error(f"Training task subprocess appears to have failed")

        return result

    except Exception as e:
        getLogger().error(f"Training task subprocess appears to have failed. {traceback.format_exc()}")
        raise


def runTestingSubprocess(config, trainingSequence, testStepIndex, generateDebugVideo=False):
    try:
        browsers = getAvailableBrowsers(config)
        windowSizes = getAvailableWindowSizes(config)

        choiceIndex = testStepIndex % (len(browsers) * len(windowSizes))
        chosenBrowser = browsers[int(choiceIndex / len(windowSizes))]
        chosenWindowSize = windowSizes[choiceIndex % len(windowSizes)]

        testingStep = TestingStep(id=str(trainingSequence.id + "_testing_step_" + str(testStepIndex)), browser=chosenBrowser, windowSize=chosenWindowSize)
        testingStep.saveToDisk(config)

        process = ManagedTaskSubprocess([sys.executable, "-m", "kwola.tasks.RunTestingStep"], {
            "config": config.serialize(),
            "testingStepId": str(testingStep.id),
            "shouldBeRandom": False,
            "generateDebugVideo": generateDebugVideo and config['enable_debug_videos'],
            "browser": chosenBrowser,
            "windowSize": chosenWindowSize
        }, timeout=config['testing_step_timeout'], config=config, logId=testingStep.id)
        process.start()
        result = process.waitForProcessResult()

        if result is not None and 'testingStepId' in result:
            result['finishTime'] = datetime.now()

            # Reload the testing sequence from the db. It will have been updated by the sub-process.
            trainingSequence.testingSteps.append(testingStep.id)
            trainingSequence.saveToDisk(config)

        getLogger().info(f"Finished the testing step {testingStep.id}")

        return result

    except Exception as e:
        getLogger().error(f"Testing task subprocess appears to have failed. {traceback.format_exc()}")
        raise

def updateModelSymbols(config, testingStepId):
    symbolMap = SymbolMapper(config)
    symbolMap.load()

    testingStep = TestingStep.loadFromDisk(testingStepId, config)

    traces = []
    totalNewSymbols = 0
    totalSplitSymbols = 0
    for executionSessionId in testingStep.executionSessions:
        executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

        for executionTraceId in executionSession.executionTraces:
            traces.append(ExecutionTrace.loadFromDisk(executionTraceId, config, applicationId=testingStep.applicationId))

        if len(traces) > 1000:
            newSymbols, splitSymbols = symbolMap.assignNewSymbols(traces)
            totalNewSymbols += newSymbols
            totalSplitSymbols += splitSymbols
            traces = []

    newSymbols, splitSymbols = symbolMap.assignNewSymbols(traces)
    totalNewSymbols += newSymbols
    totalSplitSymbols += splitSymbols
    traces = []

    getLogger().info(f"There were {totalNewSymbols} new symbols and {totalSplitSymbols} split symbols from testing step {testingStepId}")

    symbolMap.save()

def runMainTrainingLoop(config, trainingSequence, exitOnFail=False):
    stepStartTime = datetime.now()

    numberOfTrainingStepsInParallel = max(1, torch.cuda.device_count())

    chartGenerationFuture = None

    while trainingSequence.trainingLoopsCompleted < config['training_loops_needed']:
        getLogger().info(f"Starting a single training loop. Loops completed: {trainingSequence.trainingLoopsCompleted}")

        with ThreadPoolExecutor(max_workers=(config['testing_sequences_in_parallel_per_training_loop'] + numberOfTrainingStepsInParallel)) as executor:
            coordinatorTempFileName = "kwola_distributed_coordinator-" + str(random.randint(0, int(1e8)))
            coordinatorTempFilePath = os.path.join(tempfile.gettempdir(), coordinatorTempFileName)
            if os.path.exists(coordinatorTempFilePath):
                os.unlink(coordinatorTempFilePath)

            allFutures = []
            testStepFutures = []
            trainStepFutures = []

            enableDebugVideosThisLoop = bool(trainingSequence.trainingLoopsCompleted % config['debug_video_generation_frequency'] == 0)

            if torch.cuda.device_count() > 0:
                for gpu in range(numberOfTrainingStepsInParallel):
                    trainingFuture = executor.submit(runTrainingSubprocess, config, trainingSequence, trainingStepIndex=trainingSequence.trainingStepsLaunched, gpuNumber=gpu, coordinatorTempFileName=coordinatorTempFileName)
                    allFutures.append(trainingFuture)
                    trainStepFutures.append(trainingFuture)
                    trainingSequence.trainingStepsLaunched += 1
            else:
                trainingFuture = executor.submit(runTrainingSubprocess, config, trainingSequence, trainingStepIndex=trainingSequence.trainingStepsLaunched, gpuNumber=None, coordinatorTempFileName=coordinatorTempFileName)
                allFutures.append(trainingFuture)
                trainStepFutures.append(trainingFuture)
                trainingSequence.trainingStepsLaunched += 1

            for testingStepNumber in range(config['testing_sequences_per_training_loop']):
                testStepIndex = trainingSequence.testingStepsLaunched + config['training_random_initialization_sequences']
                generateDebugVideo = False
                if testingStepNumber == 0 and enableDebugVideosThisLoop:
                    generateDebugVideo = True
                future = executor.submit(runTestingSubprocess, config, trainingSequence, testStepIndex, generateDebugVideo=generateDebugVideo)
                allFutures.append(future)
                testStepFutures.append(future)
                time.sleep(3)
                trainingSequence.testingStepsLaunched += 1

            wait(allFutures)

            anyFailures = False
            for future in allFutures:
                result = future.result()
                if result is None:
                    anyFailures = True
                elif 'success' in result and not result['success']:
                    anyFailures = True

            getLogger().error(f"Finished waiting for all the result objects.")

            if anyFailures and exitOnFail:
                raise RuntimeError("One of the testing / training loops failed and did not return successfully. Exiting the training loop.")

            if not anyFailures:
                getLogger().info(f"Updating the symbols table")
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

            time.sleep(3)

        trainingSequence.trainingLoopsCompleted += 1
        trainingSequence.averageTimePerStep = (datetime.now() - stepStartTime).total_seconds() / trainingSequence.trainingLoopsCompleted
        trainingSequence.saveToDisk(config)
        if (trainingSequence.trainingLoopsCompleted % config['chart_generation_frequency']) == 0:
            enableCumulativeCoverage = bool(trainingSequence.trainingLoopsCompleted % config['chart_generate_cumulative_coverage_frequency'] == 0)
            chartGenerationFuture = AsyncThreadFuture(generateAllCharts, args=[config, None, enableCumulativeCoverage])
        getLogger().info(f"Completed one parallel training & testing step! Hooray!")

    if chartGenerationFuture is not None:
        chartGenerationFuture.wait()

    generateAllCharts(config, applicationId=None, enableCumulativeCoverage=True)

def trainAgent(config, exitOnFail=False):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    config = KwolaCoreConfiguration(config)

    # Load and save the agent to make sure all training subprocesses are synced
    agent = DeepLearningAgent(config=config, whichGpu=None)
    agent.initialize(enableTraining=False)
    agent.load()
    agent.save()
    del agent

    files = [fileName for fileName in os.listdir(config.getKwolaUserDataDirectory("training_sequences")) if ".lock" not in fileName]

    if len(files) == 0:
        trainingSequence = TrainingSequence(id=CustomIDField.generateNewUUID(TrainingSequence, config))

        trainingSequence.startTime = datetime.now()
        trainingSequence.status = "running"
        trainingSequence.trainingLoopsCompleted = 0
        trainingSequence.trainingStepsLaunched = 0
        trainingSequence.testingStepsLaunched = 0
        trainingSequence.saveToDisk(config)
    else:
        sequenceId = files[0]
        sequenceId = sequenceId.replace(".pickle", "")
        sequenceId = sequenceId.replace(".json", "")
        sequenceId = sequenceId.replace(".gz", "")

        trainingSequence = TrainingSequence.loadFromDisk(sequenceId, config)

    testingSteps = [step for step in TrainingManager.loadAllTestingSteps(config) if step.status == "completed"]

    if len(testingSteps) == 0:
        browsers = getAvailableBrowsers(config)

        # Create and destroy an environment, which forces a lot of the initial javascript in the application
        # to be loaded and translated. It also just verifies that the system can access the target URL prior
        # to trying to run a full sequence
        environment = WebEnvironment(config, sessionLimit=1, browser=browsers[0])
        environment.shutdown()
        del environment

    if len(testingSteps) < config['training_random_initialization_sequences']:
        runRandomInitialization(config, trainingSequence, exitOnFail=exitOnFail)
        trainingSequence.saveToDisk(config)

    runMainTrainingLoop(config, trainingSequence, exitOnFail=exitOnFail)

    generateAllCharts(config, enableCumulativeCoverage=True)

    trainingSequence.status = "completed"
    trainingSequence.endTime = datetime.now()
    trainingSequence.saveToDisk(config)

    for folder in config['train_agent_loop_delete_folders_on_finish']:
        fullPath = config.getKwolaUserDataDirectory(folder)
        shutil.rmtree(fullPath)
