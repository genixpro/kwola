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
from ..tasks.TaskProcess import TaskProcess
from ..config.config import Configuration
from ..datamodels.ExecutionSessionModel import ExecutionSession
from ..datamodels.BugModel import BugModel
from ..datamodels.CustomIDField import CustomIDField
from ..datamodels.TestingStepModel import TestingStep
from .RunTrainingStep import addExecutionSessionToSampleCache
from datetime import datetime
import atexit
import multiprocessing
import numpy
import os
import pickle
import tempfile
import time
import traceback


def predictedActionSubProcess(configDir, shouldBeRandom, branchFeatureSize, subProcessCommandQueue, subProcessResultQueue):
    config = Configuration(configDir)

    agent = DeepLearningAgent(config, whichGpu=None)

    agent.initialize(branchFeatureSize, enableTraining=False)
    agent.load()

    while True:
        message = subProcessCommandQueue.get()

        if message == "quit":
            break
        else:
            inferenceBatchFileName = message

        with open(inferenceBatchFileName, 'rb') as file:
            step, images, envActionMaps, additionalFeatures, pastExecutionTraces = pickle.load(file)

        os.unlink(inferenceBatchFileName)

        actions = agent.nextBestActions(step, images, envActionMaps, additionalFeatures, pastExecutionTraces, shouldBeRandom=shouldBeRandom)

        resultFileDescriptor, resultFileName = tempfile.mkstemp()
        with open(resultFileDescriptor, 'wb') as file:
            pickle.dump(actions, file)

        subProcessResultQueue.put(resultFileName)


def createDebugVideoSubProcess(configDir, branchFeatureSize, executionSessionId, name="", includeNeuralNetworkCharts=True, includeNetPresentRewardChart=True, hilightStepNumber=None, folder="debug_videos"):
    config = Configuration(configDir)

    agent = DeepLearningAgent(config, whichGpu=None)
    agent.initialize(branchFeatureSize, enableTraining=False)
    agent.load()

    kwolaDebugVideoDirectory = config.getKwolaUserDataDirectory(folder)

    executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

    videoData = agent.createDebugVideoForExecutionSession(executionSession, includeNeuralNetworkCharts=includeNeuralNetworkCharts, includeNetPresentRewardChart=includeNetPresentRewardChart, hilightStepNumber=hilightStepNumber)
    with open(os.path.join(kwolaDebugVideoDirectory, f'{name + "_" if name else ""}{str(executionSession.id)}.mp4'), "wb") as cloneFile:
        cloneFile.write(videoData)

    del agent


def loadAllBugs(config):
    bugsDir = config.getKwolaUserDataDirectory("bugs")

    bugs = []

    for fileName in os.listdir(bugsDir):
        if ".lock" not in fileName and ".txt" not in fileName and ".mp4" not in fileName:
            bugId = fileName
            bugId = bugId.replace(".json", "")
            bugId = bugId.replace(".gz", "")
            bugId = bugId.replace(".pickle", "")

            bugs.append(BugModel.loadFromDisk(bugId, config))

    return bugs


def runTestingStep(configDir, testingStepId, shouldBeRandom=False, generateDebugVideo=False):
    print(datetime.now(), f"[{os.getpid()}]", "Starting New Testing Sequence", flush=True)

    returnValue = {}

    try:
        multiprocessing.set_start_method('spawn')

        config = Configuration(configDir)

        environment = WebEnvironment(config=config)

        stepsRemaining = int(config['testing_sequence_length'])

        testStep = TestingStep.loadFromDisk(testingStepId, config)

        testStep.startTime = datetime.now()
        testStep.status = "running"
        testStep.saveToDisk(config)

        returnValue = {"testingStepId": str(testStep.id)}

        executionSessions = [
            ExecutionSession(
                id=str(testingStepId) + "_session_" + str(sessionN),
                testingStepId=str(testingStepId),
                startTime=datetime.now(),
                endTime=None,
                tabNumber=sessionN,
                executionTraces=[]
            )
            for sessionN in range(environment.numberParallelSessions())
        ]

        executionSessionTraces = [[] for sessionN in range(environment.numberParallelSessions())]

        for session in executionSessions:
            session.saveToDisk(config)

        allKnownErrorHashes = set()
        newErrorsThisTestingStep = []
        newErrorOriginalExecutionSessionIds = []
        newErrorOriginalStepNumbers = []

        bugs = loadAllBugs(config)
        for bug in bugs:
            hash = bug.error.computeHash()
            allKnownErrorHashes.add(hash)

        step = 0

        subProcesses = []

        for n in range(config['testing_subprocess_pool_size']):
            subProcessCommandQueue = multiprocessing.Queue()
            subProcessResultQueue = multiprocessing.Queue()
            subProcess = multiprocessing.Process(target=predictedActionSubProcess, args=(configDir, shouldBeRandom, environment.branchFeatureSize(), subProcessCommandQueue, subProcessResultQueue))
            subProcess.start()
            atexit.register(lambda: subProcess.terminate())

            subProcesses.append((subProcessCommandQueue, subProcessResultQueue, subProcess))

        while stepsRemaining > 0:
            stepsRemaining -= 1

            images = environment.getImages()
            envActionMaps = environment.getActionMaps()

            branchFeature = environment.getBranchFeatures()
            decayingExecutionTraceFeature = environment.getExecutionTraceFeatures()
            additionalFeatures = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=1)

            fileDescriptor, inferenceBatchFileName = tempfile.mkstemp()

            with open(fileDescriptor, 'wb') as file:
                pickle.dump((step, images, envActionMaps, additionalFeatures, executionSessionTraces), file)

            del images, envActionMaps, branchFeature, decayingExecutionTraceFeature, additionalFeatures

            subProcessCommandQueue, subProcessResultQueue, subProcess = subProcesses[0]

            subProcessCommandQueue.put(inferenceBatchFileName)
            resultFileName = subProcessResultQueue.get()
            with open(resultFileName, 'rb') as file:
                actions = pickle.load(file)
            os.unlink(resultFileName)

            if stepsRemaining % config['testing_print_every'] == 0:
                print(datetime.now(), f"[{os.getpid()}]", f"Finished {step + 1} testing actions.", flush=True)

            traces = environment.runActions(actions, [executionSession.id for executionSession in executionSessions])
            for sessionN, executionSession, trace in zip(range(len(traces)), executionSessions, traces):
                trace.executionSessionId = str(executionSession.id)
                trace.testingStepId = str(testingStepId)
                trace.saveToDisk(config)

                executionSessions[sessionN].executionTraces.append(str(trace.id))
                executionSessionTraces[sessionN].append(trace)
                executionSessions[sessionN].totalReward = float(numpy.sum(DeepLearningAgent.computePresentRewards(executionSessionTraces[sessionN], config)))

                for error in trace.errorsDetected:
                    hash = error.computeHash()

                    if hash not in allKnownErrorHashes:
                        allKnownErrorHashes.add(hash)
                        newErrorsThisTestingStep.append(error)
                        newErrorOriginalExecutionSessionIds.append(str(executionSession.id))
                        newErrorOriginalStepNumbers.append(step)

            if config['testing_reset_agent_period'] == 1 or stepsRemaining % config['testing_reset_agent_period'] == (config['testing_reset_agent_period'] - 1):
                subProcessCommandQueue, subProcessResultQueue, subProcess = subProcesses.pop(0)

                subProcessCommandQueue.put("quit")
                subProcess.terminate()

                subProcessCommandQueue = multiprocessing.Queue()
                subProcessResultQueue = multiprocessing.Queue()
                subProcess = multiprocessing.Process(target=predictedActionSubProcess, args=(configDir, shouldBeRandom, environment.branchFeatureSize(), subProcessCommandQueue, subProcessResultQueue))
                subProcess.start()
                atexit.register(lambda: subProcess.terminate())

                subProcesses.append((subProcessCommandQueue, subProcessResultQueue, subProcess))

            step += 1
            del traces
            print("", end="", sep="", flush=True)

        for subProcessCommandQueue, subProcessResultQueue, subProcess in subProcesses:
            subProcessCommandQueue.put("quit")
            subProcess.join()

        print(datetime.now(), f"[{os.getpid()}]", f"Creating movies for the execution sessions of this testing sequence.", flush=True)
        videoPaths = environment.createMovies()

        kwolaVideoDirectory = config.getKwolaUserDataDirectory("videos")

        for sessionN, videoPath, executionSession in zip(range(len(videoPaths)), videoPaths, executionSessions):
            with open(videoPath, 'rb') as origFile:
                with open(os.path.join(kwolaVideoDirectory, f'{str(executionSession.id)}.mp4'), "wb") as cloneFile:
                    cloneFile.write(origFile.read())

        totalRewards = []
        for session in executionSessions:
            print(datetime.now(), f"[{os.getpid()}]", f"Session {session.tabNumber} finished with total reward: {session.totalReward:.2f}", flush=True)
            session.saveToDisk(config)
            totalRewards.append(session.totalReward)

        print(datetime.now(), f"[{os.getpid()}]", f"Mean total reward of all sessions: ", numpy.mean(totalRewards), flush=True)

        testStep.bugsFound = len(newErrorsThisTestingStep)
        testStep.errors = newErrorsThisTestingStep

        debugVideoSubprocesses = []

        print(datetime.now(), f"[{os.getpid()}]", f"Found {len(newErrorsThisTestingStep)} new unique errors this session.", flush=True)
        for errorIndex, error, executionSessionId, stepNumber in zip(range(len(newErrorsThisTestingStep)), newErrorsThisTestingStep, newErrorOriginalExecutionSessionIds, newErrorOriginalStepNumbers):
            bug = BugModel()
            bug.id = CustomIDField.generateNewUUID(BugModel, config)
            bug.testingStepId = testStep.id
            bug.executionSessionId = executionSessionId
            bug.stepNumber = stepNumber
            bug.error = error
            bug.saveToDisk(config, overrideSaveFormat="json", overrideCompression=0)

            bugTextFile = os.path.join(config.getKwolaUserDataDirectory("bugs"), bug.id + ".txt")
            with open(bugTextFile, "wt") as file:
                file.write(bug.generateBugText())

            bugVideoFilePath = os.path.join(config.getKwolaUserDataDirectory("bugs"), bug.id + ".mp4")
            with open(os.path.join(kwolaVideoDirectory, f'{str(executionSessionId)}.mp4'), "rb") as origFile:
                with open(bugVideoFilePath, 'wb') as cloneFile:
                    cloneFile.write(origFile.read())

            debugVideoSubprocess = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, environment.branchFeatureSize(), str(executionSessionId), f"{bug.id}_bug", False, False, stepNumber, "bugs"))
            debugVideoSubprocess.start()
            atexit.register(lambda: debugVideoSubprocess.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess)

            print(datetime.now(), f"[{os.getpid()}]", f"")
            print(datetime.now(), f"[{os.getpid()}]", f"Bug #{errorIndex + 1}:")
            print(datetime.now(), f"[{os.getpid()}]", bug.generateBugText(), flush=True)
            print(datetime.now(), f"[{os.getpid()}]", f"")


        testStep.status = "completed"

        testStep.endTime = datetime.now()

        testStep.executionSessions = [session.id for session in executionSessions]
        testStep.saveToDisk(config)

        if not shouldBeRandom and generateDebugVideo:
            # Start some parallel processes generating debug videos.
            debugVideoSubprocess1 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, environment.branchFeatureSize(), str(executionSessions[0].id), "prediction", True, True, "debug_videos"))
            debugVideoSubprocess1.start()
            atexit.register(lambda: debugVideoSubprocess1.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess1)

            # Leave a gap between the two to reduce collision
            time.sleep(5)

            debugVideoSubprocess2 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, environment.branchFeatureSize(), str(executionSessions[int(len(executionSessions) / 3)].id), "mix", True, True, "debug_videos"))
            debugVideoSubprocess2.start()
            atexit.register(lambda: debugVideoSubprocess2.terminate())
            debugVideoSubprocesses.append(debugVideoSubprocess2)

        environment.shutdown()

        del environment

        for session in executionSessions:
            print(datetime.now(), f"[{os.getpid()}]", f"Preparing samples for {session.id} and adding them to the sample cache.", flush=True)
            addExecutionSessionToSampleCache(session.id, config)

        for debugVideoSubprocess in debugVideoSubprocesses:
            debugVideoSubprocess.join()

    except Exception as e:
        traceback.print_exc()
        print(datetime.now(), f"[{os.getpid()}]", "Unhandled exception occurred during testing sequence", flush=True)

    # This print statement will trigger the parent manager process to kill this process.
    print(datetime.now(), f"[{os.getpid()}]", "Finished Running Testing Sequence!", flush=True)

    return returnValue


if __name__ == "__main__":
    task = TaskProcess(runTestingStep)
    task.run()
