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
import atexit
import billiard as multiprocessing
import numpy
import os
import pickle
import tempfile
import traceback
import concurrent.futures



class TestingManager:
    def __init__(self, configDir, testingStepId, shouldBeRandom=False, generateDebugVideo=False, plugins=None):
        getLogger().info(f"[{os.getpid()}] Starting New Testing Sequence")

        self.generateDebugVideo = generateDebugVideo
        self.shouldBeRandom = shouldBeRandom
        self.configDir = configDir
        self.config = KwolaCoreConfiguration(configDir)

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
                GenerateAnnotatedVideos(self.config)
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


    def createExecutionSessions(self):
        self.executionSessions = [
            ExecutionSession(
                id=str(self.testStep.id) + "_session_" + str(sessionN),
                owner=self.testStep.owner,
                testingStepId=self.testStep.id,
                testingRunId=self.testStep.testingRunId,
                applicationId=self.testStep.applicationId,
                startTime=datetime.now(),
                endTime=None,
                tabNumber=sessionN,
                executionTraces=[]
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
            subProcess = multiprocessing.Process(target=TestingManager.predictedActionSubProcess, args=(self.configDir, self.shouldBeRandom, subProcessCommandQueue, subProcessResultQueue, preloadTraceFiles))
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
        subProcess = multiprocessing.Process(target=TestingManager.predictedActionSubProcess, args=(self.configDir, self.shouldBeRandom, subProcessCommandQueue, subProcessResultQueue, preloadTraceFiles))
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
            getLogger().warning(f"[{os.getpid()}] Removing web browser session at index {sessionToRemove} because the browser has crashed!")

            session = self.executionSessions[sessionToRemove]

            del self.executionSessions[sessionToRemove]
            del self.executionSessionTraces[sessionToRemove]
            del self.executionSessionTraceLocalPickleFiles[sessionToRemove]

            for plugin in self.plugins:
                plugin.sessionFailed(self.testStep, session)

            sessionToRemove = self.environment.removeBadSessionIfNeeded()

    def executeSingleAction(self):
        taskStartTime = datetime.now()
        images = self.environment.getImages()
        screenshotTime = (datetime.now() - taskStartTime).total_seconds()

        taskStartTime = datetime.now()
        envActionMaps = self.environment.getActionMaps()
        actionMapRetrievalTime = (datetime.now() - taskStartTime).total_seconds()

        fileDescriptor, inferenceBatchFileName = tempfile.mkstemp()

        with open(fileDescriptor, 'wb') as file:
            pickle.dump((self.step, images, envActionMaps, self.executionSessionTraceLocalPickleFiles), file)

        del images, envActionMaps

        subProcessCommandQueue, subProcessResultQueue, subProcess = self.subProcesses[0]

        taskStartTime = datetime.now()
        subProcessCommandQueue.put(inferenceBatchFileName)
        resultFileName = subProcessResultQueue.get()
        actionDecisionTime = (datetime.now() - taskStartTime).total_seconds()
        with open(resultFileName, 'rb') as file:
            actions = pickle.load(file)
        os.unlink(resultFileName)

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

            trace.executionSessionId = str(executionSession.id)
            trace.testingStepId = str(self.testStep.id)
            trace.applicationId = str(executionSession.applicationId)
            trace.testingRunId = str(executionSession.testingRunId)
            trace.owner = self.testStep.owner

            trace.timeForScreenshot = screenshotTime
            trace.timeForActionMapRetrieval = actionMapRetrievalTime
            trace.timeForActionDecision = actionDecisionTime
            trace.timeForActionExecution = actionExecutionTime
            trace.timeForMiscellaneous = miscellaneousTime

            self.executionSessions[sessionN].executionTraces.append(str(trace.id))
            self.executionSessionTraces[sessionN].append(trace)
            self.executionSessions[sessionN].totalReward = float(numpy.sum(DeepLearningAgent.computePresentRewards(self.executionSessionTraces[sessionN], self.config)))

            # We clear the actionMaps field on the trace object prior to saving the temporary pickle file. This is to reduce the amount of time
            # it takes to pickle and unpickle this object. This is a bit of a HACK and depends on the fact that the DeepLearningAgent.nextBestActions
            # function does not actually require this field at all. Without this, the pickling/unpickling can come to take up to 90% of the total time
            # of each loop
            actionMaps = trace.actionMaps
            trace.actionMaps = None
            fileDescriptor, traceFileName = tempfile.mkstemp()
            with open(fileDescriptor, 'wb') as file:
                pickle.dump(trace, file)
            self.executionSessionTraceLocalPickleFiles[sessionN].append(traceFileName)
            trace.actionMaps = actionMaps

            # Submit a lambda to save this trace to disk. This is done in the background to avoid
            # holding up the main loop. Saving the trace to disk can be time consuming.
            self.traceSaveExecutor.submit(TestingManager.saveTrace, trace, self.config)

        if len(validTraces) > 0:
            for plugin in self.testingStepPlugins:
                plugin.afterActionsRun(self.testStep, validTracePairedExecutionSessions, validTraces)

        del traces

    def savePlainVideoFiles(self):
        getLogger().info(f"[{os.getpid()}] Creating movies for the execution sessions of this testing sequence.")

        moviePlugin = [plugin for plugin in self.environment.plugins if isinstance(plugin, RecordScreenshots)][0]
        videoPaths = [moviePlugin.movieFilePath(executionSession) for executionSession in self.executionSessions]

        kwolaVideoDirectory = self.config.getKwolaUserDataDirectory("videos")

        for executionSession, sessionN, videoPath in zip(self.executionSessions, range(len(videoPaths)), videoPaths):
            with open(videoPath, 'rb') as origFile:
                with open(os.path.join(kwolaVideoDirectory, f'{str(executionSession.id)}.mp4'), "wb") as cloneFile:
                    cloneFile.write(origFile.read())


    def shutdownEnvironment(self):
        self.environment.shutdown()
        del self.environment


    @staticmethod
    def predictedActionSubProcess(configDir, shouldBeRandom, subProcessCommandQueue, subProcessResultQueue, preloadTraceFiles):
        setupLocalLogging()

        config = KwolaCoreConfiguration(configDir)

        agent = DeepLearningAgent(config, whichGpu=None)

        agent.initialize(enableTraining=False)
        agent.load()

        loadedPastExecutionTraces = {}

        def preloadFile(fileName):
            nonlocal loadedPastExecutionTraces
            with open(fileName, 'rb') as file:
                loadedPastExecutionTraces[fileName] = pickle.load(file)

        with concurrent.futures.ThreadPoolExecutor(max_workers=config['testing_trace_load_workers']) as loadExecutor:
            for fileName in preloadTraceFiles:
                loadExecutor.submit(preloadFile, fileName)

        while True:
            message = subProcessCommandQueue.get()

            if message == "quit":
                break
            else:
                inferenceBatchFileName = message

            with open(inferenceBatchFileName, 'rb') as file:
                step, images, envActionMaps, pastExecutionTraceLocalTempFiles = pickle.load(file)

            pastExecutionTraces = [[] for n in range(len(pastExecutionTraceLocalTempFiles))]
            for sessionN, traceFileNameList in enumerate(pastExecutionTraceLocalTempFiles):
                for fileName in traceFileNameList:
                    if fileName in loadedPastExecutionTraces:
                        pastExecutionTraces[sessionN].append(loadedPastExecutionTraces[fileName])
                    else:
                        with open(fileName, 'rb') as file:
                            trace = pickle.load(file)
                            pastExecutionTraces[sessionN].append(trace)
                            loadedPastExecutionTraces[fileName] = trace


            os.unlink(inferenceBatchFileName)

            actions = agent.nextBestActions(step, images, envActionMaps, pastExecutionTraces, shouldBeRandom=shouldBeRandom)

            resultFileDescriptor, resultFileName = tempfile.mkstemp()
            with open(resultFileDescriptor, 'wb') as file:
                pickle.dump(actions, file)

            subProcessResultQueue.put(resultFileName)

    @staticmethod
    def saveTrace(trace, config):
        trace.saveToDisk(config)

    def runTesting(self):
        getLogger().info(f"[{os.getpid()}] Starting New Testing Sequence")

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
            self.createTestingSubprocesses()

            for plugin in self.testingStepPlugins:
                plugin.testingStepStarted(self.testStep, self.executionSessions)

            self.environment = WebEnvironment(config=self.config, executionSessions=self.executionSessions)

            self.loopTime = datetime.now()
            while self.stepsRemaining > 0:
                self.stepsRemaining -= 1

                self.removeBadSessions()
                if len(self.executionSessions) == 0:
                    break

                self.executeSingleAction()

                if self.config['testing_reset_agent_period'] == 1 or self.stepsRemaining % self.config['testing_reset_agent_period'] == (self.config['testing_reset_agent_period'] - 1):
                    self.restartOneTestingSubprocess()

                self.step += 1

            self.killAndJoinTestingSubprocesses()
            self.removeBadSessions()

            # Ensure all the trace objects get saved to disc
            self.traceSaveExecutor.shutdown()

            self.environment.runSessionCompletedHooks()

            self.savePlainVideoFiles()

            for session in self.executionSessions:
                session.endTime = datetime.now()
                session.saveToDisk(self.config)

            # We shutdown the environment before generating the annotated videos in order
            # to conserve memory, since the environment is no longer needed after this point
            self.shutdownEnvironment()

            self.testStep.status = "completed"
            self.testStep.endTime = datetime.now()
            self.testStep.executionSessions = [session.id for session in self.executionSessions]

            for plugin in self.testingStepPlugins:
                plugin.testingStepFinished(self.testStep, self.executionSessions)

            self.testStep.saveToDisk(self.config)
            resultValue['successfulExecutionSessions'] = len(self.testStep.executionSessions)
            resultValue['success'] = True

            for traceList in self.executionSessionTraceLocalPickleFiles:
                for fileName in traceList:
                    os.unlink(fileName)

        except Exception as e:
            getLogger().error(f"[{os.getpid()}] Unhandled exception occurred during testing sequence:\n{traceback.format_exc()}")
            resultValue['success'] = False
            resultValue['exception'] = traceback.format_exc()

        # This print statement will trigger the parent manager process to kill this process.
        getLogger().info(f"[{os.getpid()}] Finished Running Testing Sequence!")

        return resultValue
