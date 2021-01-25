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


from ...components.agents.DeepLearningAgent import DeepLearningAgent
from ...components.environments.WebEnvironment import WebEnvironment
from ...config.config import KwolaCoreConfiguration
from ...config.logger import getLogger, setupLocalLogging
from ...datamodels.ExecutionSessionModel import ExecutionSession
from ...datamodels.ExecutionTraceModel import ExecutionTrace
from ...datamodels.TestingStepModel import TestingStep
from ..plugins.core.CreateLocalBugObjects import CreateLocalBugObjects
from datetime import datetime
from kwola.components.plugins.base.TestingStepPluginBase import TestingStepPluginBase
from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
from kwola.components.plugins.core.GenerateAnnotatedVideos import GenerateAnnotatedVideos
from kwola.components.plugins.core.GenerateDebugVideos import GenerateDebugVideos
from kwola.components.plugins.core.LogSessionActionExecutionTimes import LogSessionActionExecutionTimes
from kwola.components.plugins.core.LogSessionRewards import LogSessionRewards
from kwola.components.plugins.core.PrecomputeSessionsForSampleCache import PrecomputeSessionsForSampleCache
from kwola.components.plugins.core.RecordScreenshots import RecordScreenshots
from kwola.errors import ProxyVerificationFailed
import atexit
import billiard as multiprocessing
import numpy
import os
import pickle
import tempfile
import traceback
import selenium.common.exceptions
import concurrent.futures
from kwola.components.utils.retry import autoretry
from pprint import pformat



class TestingStepManager:
    def __init__(self, config, testingStepId, shouldBeRandom=False, generateDebugVideo=False, plugins=None, browser=None, windowSize=None):
        getLogger().info(f"Starting New Testing Sequence")

        self.generateDebugVideo = generateDebugVideo
        self.shouldBeRandom = shouldBeRandom
        self.config = KwolaCoreConfiguration(config)
        self.browser = browser
        self.windowSize = windowSize

        self.environment = None

        self.stepsRemaining = int(self.config['testing_sequence_length'])

        self.testStep = TestingStep.loadFromDisk(testingStepId, self.config)

        self.executionSessions = []
        self.executionSessionTraces = []
        self.executionSessionTraceLocalPickleFiles = []

        self.step = 0

        if plugins is None:
            self.plugins = [
                CreateLocalBugObjects(self.config),
                LogSessionRewards(self.config),
                LogSessionActionExecutionTimes(self.config),
                PrecomputeSessionsForSampleCache(self.config),
                # GenerateAnnotatedVideos(self.config)
            ]

            if not shouldBeRandom and generateDebugVideo:
                self.plugins.append(GenerateDebugVideos(self.config))
        else:
            self.plugins = plugins

        self.webEnvironmentPlugins = [
            plugin for plugin in self.plugins
            if isinstance(plugin, WebEnvironmentPluginBase)
        ]

        self.testingStepPlugins = [
            plugin for plugin in self.plugins
            if isinstance(plugin, TestingStepPluginBase)
        ]

        self.traceSaveExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config['testing_trace_save_workers'])

        self.agent = DeepLearningAgent(self.config, whichGpu=None)
        if not self.config['testing_enable_prediction_subprocess']:
            self.agent.initialize(enableTraining=False)
            try:
                self.agent.load()
            except RuntimeError as e:
                getLogger().error(
                    f"Warning! DeepLearningAgent was unable to load the model file from disk, and so is instead using a freshly random initialized neural network. The original error is: {traceback.format_exc()}")
                self.agent.save()

    def finishAllTraceSaves(self):
        executor = self.traceSaveExecutor
        self.traceSaveExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=self.config['testing_trace_save_workers'])
        executor.shutdown()

    def createExecutionSessions(self):
        self.executionSessions = [
            ExecutionSession(
                id=str(self.testStep.id) + "-session-" + str(sessionN),
                owner=self.testStep.owner,
                status="running",
                testingStepId=self.testStep.id,
                testingRunId=self.testStep.testingRunId,
                applicationId=self.testStep.applicationId,
                startTime=datetime.now(),
                endTime=None,
                tabNumber=sessionN,
                executionTraces=[],
                browser=self.browser,
                windowSize=self.windowSize
            )
            for sessionN in range(self.config['web_session_parallel_execution_sessions'])
        ]

        self.executionSessionTraces = [[] for sessionN in range(self.config['web_session_parallel_execution_sessions'])]
        self.executionSessionTraceLocalPickleFiles = [[] for sessionN in range(self.config['web_session_parallel_execution_sessions'])]

        for session in self.executionSessions:
            session.saveToDisk(self.config)


    def createTestingSubprocesses(self):
        self.subProcesses = []

        for n in range(self.config['testing_subprocess_pool_size']):
            subProcessCommandQueue = multiprocessing.Queue()
            subProcessResultQueue = multiprocessing.Queue()
            preloadTraceFiles = [file for fileList in self.executionSessionTraceLocalPickleFiles for file in fileList]
            subProcess = multiprocessing.Process(target=TestingStepManager.predictedActionSubProcess, args=(self.config.serialize(), self.shouldBeRandom, subProcessCommandQueue, subProcessResultQueue, preloadTraceFiles))
            subProcess.start()
            atexit.register(lambda: subProcess.terminate())

            self.subProcesses.append((subProcessCommandQueue, subProcessResultQueue, subProcess))

    def restartOneTestingSubprocess(self):
        subProcessCommandQueue, subProcessResultQueue, subProcess = self.subProcesses.pop(0)

        subProcessCommandQueue.put("quit")
        subProcess.terminate()

        subProcessCommandQueue = multiprocessing.Queue()
        subProcessResultQueue = multiprocessing.Queue()
        preloadTraceFiles = [file for fileList in self.executionSessionTraceLocalPickleFiles for file in fileList]
        subProcess = multiprocessing.Process(target=TestingStepManager.predictedActionSubProcess, args=(self.config.serialize(), self.shouldBeRandom, subProcessCommandQueue, subProcessResultQueue, preloadTraceFiles))
        subProcess.start()
        atexit.register(lambda: subProcess.terminate())

        self.subProcesses.append((subProcessCommandQueue, subProcessResultQueue, subProcess))

    def killAndJoinTestingSubprocesses(self):
        for subProcessCommandQueue, subProcessResultQueue, subProcess in self.subProcesses:
            subProcessCommandQueue.put("quit")
            subProcess.join()

    def removeBadSessions(self):
        sessionToRemove = self.environment.removeBadSessionIfNeeded()
        while sessionToRemove is not None:
            session = self.executionSessions[sessionToRemove]
            session.status = "failed"
            session.endTime = datetime.now()

            del self.executionSessions[sessionToRemove]
            del self.executionSessionTraces[sessionToRemove]
            del self.executionSessionTraceLocalPickleFiles[sessionToRemove]

            for plugin in self.testingStepPlugins:
                plugin.sessionFailed(self.testStep, session)

            session.saveToDisk(self.config)

            sessionToRemove = self.environment.removeBadSessionIfNeeded()

    def executeSingleAction(self):
        taskStartTime = datetime.now()
        images = self.environment.getImages()
        screenshotTime = (datetime.now() - taskStartTime).total_seconds()

        taskStartTime = datetime.now()
        envActionMaps = self.environment.getActionMaps()
        actionMapRetrievalTime = (datetime.now() - taskStartTime).total_seconds()

        if self.config['testing_enable_prediction_subprocess']:
            fileDescriptor, inferenceBatchFileName = tempfile.mkstemp()

            with open(fileDescriptor, 'wb') as file:
                pickle.dump((self.step, images, envActionMaps, self.executionSessionTraceLocalPickleFiles), file, protocol=pickle.HIGHEST_PROTOCOL)

            del images, envActionMaps
            subProcessCommandQueue, subProcessResultQueue, subProcess = self.subProcesses[0]

            taskStartTime = datetime.now()
            subProcessCommandQueue.put(inferenceBatchFileName)
            resultFileName = subProcessResultQueue.get()
            with open(resultFileName, 'rb') as file:
                actions = pickle.load(file)
            os.unlink(resultFileName)
            actionDecisionTime = (datetime.now() - taskStartTime).total_seconds()
        else:
            taskStartTime = datetime.now()
            # We have to make sure all trace saves are finished before nextBestActions because there is a multithreading
            # problem in ExecutionTrace.saveToDisk due to the way it handles the cached branch trace objects.
            self.finishAllTraceSaves()
            actions, times = self.agent.nextBestActions(self.step, images, envActionMaps, self.executionSessionTraces, shouldBeRandom=self.shouldBeRandom)
            actionDecisionTime = (datetime.now() - taskStartTime).total_seconds()

            if actionDecisionTime > 10.0:
                msg = f"Finished agent.nextBestActions after {actionDecisionTime} seconds. Subtimes:"
                for key, time in times.items():
                    msg += f"\n    {key}: {time:.5f}"
                getLogger().info(msg)

        for plugin in self.testingStepPlugins:
            plugin.beforeActionsRun(self.testStep, self.executionSessions, actions)

        taskStartTime = datetime.now()
        traces = self.environment.runActions(actions)
        actionExecutionTime = (datetime.now() - taskStartTime).total_seconds()

        totalLoopTime = (datetime.now() - self.loopTime).total_seconds()
        self.loopTime = datetime.now()

        miscellaneousTime = totalLoopTime - (screenshotTime + actionMapRetrievalTime + actionDecisionTime + actionExecutionTime)

        validTraces = []
        validTracePairedExecutionSessions = []

        for sessionN, executionSession, trace in zip(range(len(traces)), self.executionSessions, traces):
            if trace is None:
                continue

            validTraces.append(trace)
            validTracePairedExecutionSessions.append(executionSession)

            trace.timeForScreenshot = screenshotTime
            trace.timeForActionMapRetrieval = actionMapRetrievalTime
            trace.timeForActionDecision = actionDecisionTime
            trace.timeForActionExecution = actionExecutionTime
            trace.timeForMiscellaneous = miscellaneousTime

            self.executionSessions[sessionN].executionTraces.append(str(trace.id))
            self.executionSessionTraces[sessionN].append(trace)
            self.executionSessions[sessionN].totalReward = float(numpy.sum(DeepLearningAgent.computePresentRewards(self.executionSessionTraces[sessionN], self.config)))

            self.agent.symbolMapper.computeCachedCumulativeBranchTraces(self.executionSessionTraces[sessionN])
            self.agent.symbolMapper.computeCachedDecayingBranchTrace(self.executionSessionTraces[sessionN])
            # Don't need to compute the future branch trace since it is only used in training and not at inference time
            # self.agent.symbolMapper.computeCachedDecayingFutureBranchTrace(self.executionSessionTraces[sessionN])

            if self.config['testing_enable_prediction_subprocess']:
                # We clear the actionMaps field on the trace object prior to saving the temporary pickle file. This is to reduce the amount of time
                # it takes to pickle and unpickle this object. This is a bit of a HACK and depends on the fact that the DeepLearningAgent.nextBestActions
                # function does not actually require this field at all. Without this, the pickling/unpickling can come to take up to 90% of the total time
                # of each loop
                actionMaps = trace.actionMaps
                trace.actionMaps = None
                fileDescriptor, traceFileName = tempfile.mkstemp()
                with open(fileDescriptor, 'wb') as file:
                    # file.write(trace.to_json())
                    pickle.dump(trace, file, protocol=pickle.HIGHEST_PROTOCOL)
                self.executionSessionTraceLocalPickleFiles[sessionN].append(traceFileName)
                trace.actionMaps = actionMaps

        if len(validTraces) > 0:
            for plugin in self.testingStepPlugins:
                plugin.afterActionsRun(self.testStep, validTracePairedExecutionSessions, validTraces)

        del traces

    def shutdownEnvironment(self):
        self.environment.shutdown()
        del self.environment


    @staticmethod
    def predictedActionSubProcess(config, shouldBeRandom, subProcessCommandQueue, subProcessResultQueue, preloadTraceFiles):
        setupLocalLogging(config)

        config = KwolaCoreConfiguration(config)

        agent = DeepLearningAgent(config, whichGpu=None)

        agent.initialize(enableTraining=False)
        try:
            agent.load()
        except RuntimeError as e:
            getLogger().error(f"Warning! DeepLearningAgent was unable to load the model file from disk, and so is instead using a freshly random initialized neural network. The original error is: {traceback.format_exc()}")
            agent.save()

        loadedPastExecutionTraces = {}

        def preloadFile(fileName):
            nonlocal loadedPastExecutionTraces
            fileData = config.loadKwolaFileData("execution_traces", fileName)
            loadedPastExecutionTraces[fileName] = pickle.loads(fileData)

        preloadStartTime = datetime.now()
        with concurrent.futures.ThreadPoolExecutor(max_workers=config['testing_trace_load_workers']) as loadExecutor:
            for fileName in preloadTraceFiles:
                loadExecutor.submit(preloadFile, fileName)
        preloadTime = (datetime.now() - preloadStartTime).total_seconds()
        if preloadTime > 5:
            getLogger().info(f"Preloaded {len(preloadTraceFiles)} traces. Finished preload after {preloadTime} seconds")

        while True:
            message = subProcessCommandQueue.get()

            if message == "quit":
                break
            else:
                inferenceBatchFileName = message

            traceLoadStartTime = datetime.now()

            with open(inferenceBatchFileName, 'rb') as file:
                step, images, envActionMaps, pastExecutionTraceLocalTempFiles = pickle.load(file)

            pastExecutionTraces = [[] for n in range(len(pastExecutionTraceLocalTempFiles))]
            for sessionN, traceFileNameList in enumerate(pastExecutionTraceLocalTempFiles):
                for fileName in traceFileNameList:
                    if fileName in loadedPastExecutionTraces:
                        pastExecutionTraces[sessionN].append(loadedPastExecutionTraces[fileName])
                    else:
                        fileData = config.loadKwolaFileData("execution_traces", fileName)
                        trace = pickle.loads(fileData)
                        pastExecutionTraces[sessionN].append(trace)
                        loadedPastExecutionTraces[fileName] = trace


            os.unlink(inferenceBatchFileName)

            traceLoadTime = (datetime.now() - traceLoadStartTime).total_seconds()
            if traceLoadTime > 5:
                getLogger().info(f"Finished trace load after {traceLoadTime} seconds")

            nextBestActionsStartTime = datetime.now()
            actions, times = agent.nextBestActions(step, images, envActionMaps, pastExecutionTraces, shouldBeRandom=shouldBeRandom)

            nextBestActionsTime = (datetime.now() - nextBestActionsStartTime).total_seconds()
            if nextBestActionsTime > 5:
                msg = f"Finished agent.nextBestActions after {nextBestActionsTime} seconds. Subtimes:"
                for key, time in times.items():
                    msg += f"\n    {key}: {time:.5f}"
                getLogger().info(msg)

            resultFileDescriptor, resultFileName = tempfile.mkstemp()
            with open(resultFileDescriptor, 'wb') as file:
                pickle.dump(actions, file, protocol=pickle.HIGHEST_PROTOCOL)

            subProcessResultQueue.put(resultFileName)

    @staticmethod
    @autoretry()
    def saveTrace(trace, config):
        trace.saveToDisk(config)

    def runTesting(self):
        if self.config['print_configuration_on_startup']:
            getLogger().info(f"Starting New Testing Sequence with configuration:\n{pformat(self.config.configData)}")
        else:
            getLogger().info(f"Starting New Testing Sequence")

        resultValue = {'success': True}

        try:
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass

            self.testStep.startTime = datetime.now()
            self.testStep.status = "running"
            self.testStep.saveToDisk(self.config)

            resultValue["testingStepId"] = str(self.testStep.id)

            self.createExecutionSessions()
            if self.config['testing_enable_prediction_subprocess']:
                self.createTestingSubprocesses()

            for plugin in self.testingStepPlugins:
                plugin.testingStepStarted(self.testStep, self.executionSessions)

            self.environment = WebEnvironment(config=self.config, executionSessions=self.executionSessions, plugins=self.webEnvironmentPlugins, browser=self.browser, windowSize=self.windowSize)

            self.loopTime = datetime.now()
            while self.stepsRemaining > 0:
                self.stepsRemaining -= 1

                self.removeBadSessions()
                if len(self.executionSessions) == 0:
                    break

                self.executeSingleAction()

                if self.config['testing_enable_prediction_subprocess']:
                    if self.config['testing_reset_agent_period'] == 1 or self.stepsRemaining % self.config['testing_reset_agent_period'] == (self.config['testing_reset_agent_period'] - 1):
                        self.restartOneTestingSubprocess()

                self.step += 1

            if self.config['testing_enable_prediction_subprocess']:
                self.killAndJoinTestingSubprocesses()
            self.removeBadSessions()

            # Compute the code prevalence scores for all of the traces
            allTraces = [trace for traceList in self.executionSessionTraces for trace in traceList]
            self.agent.symbolMapper.load()
            self.agent.symbolMapper.computeCodePrevalenceScores(allTraces)

            # Ensure all the trace objects get saved to disc
            for trace in allTraces:
                self.traceSaveExecutor.submit(TestingStepManager.saveTrace, trace, self.config)

            self.traceSaveExecutor.shutdown()

            self.environment.runSessionCompletedHooks()

            for session in self.executionSessions:
                session.endTime = datetime.now()
                session.saveToDisk(self.config)

            # We shutdown the environment before generating the annotated videos in order
            # to conserve memory, since the environment is no longer needed after this point
            self.shutdownEnvironment()

            if len(self.executionSessions) == 0:
                self.testStep.status = "failed"
            else:
                self.testStep.status = "completed"
                self.testStep.browser = self.browser
                self.testStep.userAgent = self.executionSessions[0].userAgent

            self.testStep.endTime = datetime.now()
            self.testStep.executionSessions = [session.id for session in self.executionSessions]

            if len(self.executionSessions) > 0:
                for plugin in self.testingStepPlugins:
                    plugin.testingStepFinished(self.testStep, self.executionSessions)

            for session in self.executionSessions:
                session.status = "completed"
                session.saveToDisk(self.config)

            self.testStep.saveToDisk(self.config)
            resultValue['successfulExecutionSessions'] = len(self.testStep.executionSessions)
            if len(self.testStep.executionSessions) == 0:
                resultValue['success'] = False
            else:
                resultValue['success'] = True

            for traceList in self.executionSessionTraceLocalPickleFiles:
                for fileName in traceList:
                    os.unlink(fileName)

        except selenium.common.exceptions.WebDriverException:
            # This error just happens sometimes. It has something to do with the chrome process failing to interact correctly
            # with mitmproxy. Its not at all clear what causes it, but the system can't auto retry from it unless the whole container
            # is killed. So we just explicitly catch it here so we don't trigger an error level log message, which gets sent to slack.
            # The manager process will safely restart this testing step.
            getLogger().warning(f"Unhandled exception occurred during testing sequence:\n{traceback.format_exc()}")
            resultValue['success'] = False
            resultValue['exception'] = traceback.format_exc()
        except ProxyVerificationFailed:
            # Handle this errors gracefully without an error level message. This happens more often when our own servers go down
            # then when the proxy is actually not functioning
            getLogger().warning(f"Unhandled exception occurred during testing sequence:\n{traceback.format_exc()}")
            resultValue['success'] = False
            resultValue['exception'] = traceback.format_exc()
        except Exception as e:
            getLogger().error(f"Unhandled exception occurred during testing sequence:\n{traceback.format_exc()}")
            resultValue['success'] = False
            resultValue['exception'] = traceback.format_exc()

        # This print statement will trigger the parent manager process to kill this process.
        getLogger().info(f"Finished Running Testing Sequence!")

        return resultValue
