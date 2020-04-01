from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.TestingStepModel import TestingStep
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.components.TaskProcess import TaskProcess
from kwola.config.config import Configuration
import pickle
from datetime import datetime
import time
import numpy
import traceback
import tempfile
import os
import multiprocessing
import atexit


def predictedActionSubProcess(configDir, shouldBeRandom, branchFeatureSize, subProcessCommandQueue, subProcessResultQueue):
    config = Configuration(configDir)

    agent = DeepLearningAgent(config, whichGpu=None)

    agent.initialize(branchFeatureSize)
    agent.load()

    while True:
        message = subProcessCommandQueue.get()

        if message == "quit":
            break
        else:
            inferenceBatchFileName = message


        with open(inferenceBatchFileName, 'rb') as file:
            step, images, envActionMaps, additionalFeatures, lastActions = pickle.load(file)

        os.unlink(inferenceBatchFileName)

        actions = agent.nextBestActions(step, images, envActionMaps, additionalFeatures, lastActions, shouldBeRandom=shouldBeRandom)

        resultFileDescriptor, resultFileName = tempfile.mkstemp()
        with open(resultFileDescriptor, 'wb') as file:
            pickle.dump(actions, file)

        subProcessResultQueue.put(resultFileName)


def createDebugVideoSubProcess(configDir, branchFeatureSize, executionSessionId, name=""):
    config = Configuration(configDir)

    agent = DeepLearningAgent(config, whichGpu=None)
    agent.initialize(branchFeatureSize)
    agent.load()

    kwolaDebugVideoDirectory = config.getKwolaUserDataDirectory("debug_videos")

    executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

    videoData = agent.createDebugVideoForExecutionSession(executionSession)
    with open(os.path.join(kwolaDebugVideoDirectory, f'{name + "_" if name else ""}{str(executionSession.id)}.mp4'), "wb") as cloneFile:
        cloneFile.write(videoData)

    del agent


def runTestingStep(configDir, testingStepId, shouldBeRandom=False, generateDebugVideo=False):
    print(datetime.now(), "Starting New Testing Sequence", flush=True)

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

        for session in executionSessions:
            session.saveToDisk(config)

        errorHashes = set()
        uniqueErrors = []

        step = 0

        subProcesses = []

        for n in range(config['testing_subprocess_pool_size']):
            subProcessCommandQueue = multiprocessing.Queue()
            subProcessResultQueue = multiprocessing.Queue()
            subProcess = multiprocessing.Process(target=predictedActionSubProcess, args=(configDir, shouldBeRandom, environment.branchFeatureSize(), subProcessCommandQueue, subProcessResultQueue))
            subProcess.start()
            atexit.register(lambda: subProcess.terminate())

            subProcesses.append((subProcessCommandQueue, subProcessResultQueue, subProcess))

        recentActions = [[] for n in range(environment.numberParallelSessions())]

        while stepsRemaining > 0:
            stepsRemaining -= 1

            images = environment.getImages()
            envActionMaps = environment.getActionMaps()

            branchFeature = environment.getBranchFeatures()
            decayingExecutionTraceFeature = environment.getExecutionTraceFeatures()
            additionalFeatures = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=1)

            fileDescriptor, inferenceBatchFileName = tempfile.mkstemp()

            with open(fileDescriptor, 'wb') as file:
                pickle.dump((step, images, envActionMaps, additionalFeatures, recentActions), file)

            del images, envActionMaps, branchFeature, decayingExecutionTraceFeature, additionalFeatures

            subProcessCommandQueue, subProcessResultQueue, subProcess = subProcesses[0]

            subProcessCommandQueue.put(inferenceBatchFileName)
            resultFileName = subProcessResultQueue.get()
            with open(resultFileName, 'rb') as file:
                actions = pickle.load(file)
            os.unlink(resultFileName)

            if stepsRemaining % config['testing_print_every'] == 0:
                print(datetime.now(), f"Finished {step + 1} testing actions.", flush=True)

            step += 1

            traces = environment.runActions(actions, [executionSession.id for executionSession in executionSessions])
            for sessionN, executionSession, trace in zip(range(len(traces)), executionSessions, traces):
                trace.executionSessionId = str(executionSession.id)
                trace.testingStepId = str(testingStepId)
                trace.saveToDisk(config)

                executionSessions[sessionN].executionTraces.append(str(trace.id))
                executionSessions[sessionN].totalReward = float(numpy.sum(DeepLearningAgent.computePresentRewards(executionSessions[sessionN], config)))

                for error in trace.errorsDetected:
                    hash = error.computeHash()

                    if hash not in errorHashes:
                        errorHashes.add(hash)
                        uniqueErrors.append(error)

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

            for sampleRecentActions, action, trace in zip(recentActions, actions, traces):
                # Clear the recent actions list every time new branches of code get executed.
                if trace.didNewBranchesExecute:
                    sampleRecentActions.clear()

                sampleRecentActions.append(action)

            del traces
            print("", end="", sep="", flush=True)


        for subProcessCommandQueue, subProcessResultQueue, subProcess in subProcesses:
            subProcessCommandQueue.put("quit")
            subProcess.join()

        print(datetime.now(), f"Creating movies for the execution sessions of this testing sequence.", flush=True)
        videoPaths = environment.createMovies()

        kwolaVideoDirectory = config.getKwolaUserDataDirectory("videos")

        for sessionN, videoPath, executionSession in zip(range(len(videoPaths)), videoPaths, executionSessions):
            with open(videoPath, 'rb') as origFile:
                with open(os.path.join(kwolaVideoDirectory, f'{str(executionSession.id)}.mp4'), "wb") as cloneFile:
                    cloneFile.write(origFile.read())

        totalRewards = []
        for session in executionSessions:
            print(datetime.now(), f"Session {session.tabNumber} finished with total reward: {session.totalReward:.2f}", flush=True)
            session.saveToDisk(config)
            totalRewards.append(session.totalReward)

        print(datetime.now(), f"Mean total reward of all sessions: ", numpy.mean(totalRewards), flush=True)

        testStep.bugsFound = len(uniqueErrors)
        testStep.errors = uniqueErrors

        testStep.status = "completed"

        testStep.endTime = datetime.now()

        testStep.executionSessions = [session.id for session in executionSessions]
        testStep.saveToDisk(config)

        if not shouldBeRandom and generateDebugVideo:
            # Start some parallel processes generating debug videos.
            debugVideoSubprocess1 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, environment.branchFeatureSize(), str(executionSessions[0].id), "prediction"))
            debugVideoSubprocess1.start()
            atexit.register(lambda: debugVideoSubprocess1.terminate())

            # Leave a gap between the two to reduce collision
            time.sleep(5)

            debugVideoSubprocess2 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(configDir, environment.branchFeatureSize(), str(executionSessions[int(len(executionSessions) / 3)].id), "mix"))
            debugVideoSubprocess2.start()
            atexit.register(lambda: debugVideoSubprocess2.terminate())

            debugVideoSubprocess1.join()
            debugVideoSubprocess2.join()

        environment.shutdown()

        del environment
    except Exception as e:
        traceback.print_exc()
        print(datetime.now(), "Unhandled exception occurred during testing sequence", flush=True)

    # This print statement will trigger the parent manager process to kill this process.
    print(datetime.now(), "Finished Running Testing Sequence!", flush=True)

    return returnValue


if __name__ == "__main__":
    task = TaskProcess(runTestingStep)
    task.run()

