from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.ExecutionTraceModel import ExecutionTrace
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.components.agents.RandomAgent import RandomAgent
from kwola.components.TaskProcess import TaskProcess
import random
import pickle
from datetime import datetime
import pandas as pd
import torch
import numpy
import concurrent.futures
import traceback
from bson import ObjectId
import tempfile
import os
import multiprocessing
from multiprocessing import Pool
from kwola.config import config

def predictedActionSubProcess(shouldBeRandom, branchFeatureSize, subProcessCommandQueue, subProcessResultQueue):
    if shouldBeRandom:
        agent = RandomAgent()
    else:
        agent = DeepLearningAgent(config.getAgentConfiguration(), whichGpu=None)

    agent.initialize(branchFeatureSize)
    agent.load()

    while True:
        message = subProcessCommandQueue.get()

        if message == "quit":
            break
        else:
            inferenceBatchFileName = message


        with open(inferenceBatchFileName, 'rb') as file:
            step, images, envActionMaps, additionalFeatures = pickle.load(file)

        os.unlink(inferenceBatchFileName)

        actions = agent.nextBestActions(step, images, envActionMaps, additionalFeatures)

        resultFileDescriptor, resultFileName = tempfile.mkstemp()
        with open(resultFileDescriptor, 'wb') as file:
            pickle.dump(actions, file)

        subProcessResultQueue.put(resultFileName)


def runTestingSequence(testingSequenceId, shouldBeRandom=False):
    print("Starting New Testing Sequence", flush=True)

    returnValue = {}

    try:
        agentConfig = config.getAgentConfiguration()

        environment = WebEnvironment(environmentConfiguration=config.getWebEnvironmentConfiguration())

        stepsRemaining = int(agentConfig['testing_sequence_length'])

        testSequence = TestingSequenceModel.objects(id=testingSequenceId).first()

        testSequence.startTime = datetime.now()
        testSequence.status = "running"
        testSequence.save()

        returnValue = {"testingSequenceId": str(testSequence.id)}

        executionSessions = [
            ExecutionSession(
                testingSequenceId=str(testingSequenceId),
                startTime=datetime.now(),
                endTime=None,
                tabNumber=sessionN,
                executionTraces=[]
            )
            for sessionN in range(environment.numberParallelSessions())
        ]

        for session in executionSessions:
            session.save()

        errorHashes = set()
        uniqueErrors = []

        step = 0

        subProcessCommandQueue = multiprocessing.Queue()
        subProcessResultQueue = multiprocessing.Queue()
        subProcess = multiprocessing.Process(target=predictedActionSubProcess, args=(shouldBeRandom, environment.branchFeatureSize(), subProcessCommandQueue, subProcessResultQueue))
        subProcess.start()

        while stepsRemaining > 0:
            stepsRemaining -= 1

            images = environment.getImages()
            envActionMaps = environment.getActionMaps()

            branchFeature = environment.getBranchFeatures()
            decayingExecutionTraceFeature = environment.getExecutionTraceFeatures()
            additionalFeatures = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=1)

            fileDescriptor, inferenceBatchFileName = tempfile.mkstemp()

            with open(fileDescriptor, 'wb') as file:
                pickle.dump((step, images, envActionMaps, additionalFeatures), file)

            subProcessCommandQueue.put(inferenceBatchFileName)
            resultFileName = subProcessResultQueue.get()
            with open(resultFileName, 'rb') as file:
                actions = pickle.load(file)
            os.unlink(resultFileName)

            step += 1

            traces = environment.runActions(actions)
            for sessionN, executionSession, trace in zip(range(len(traces)), executionSessions, traces):
                trace.executionSessionId = str(executionSession.id)
                trace.testingSequenceId = str(testingSequenceId)
                trace.save()

                executionSessions[sessionN].executionTraces.append(trace)

                for error in trace.errorsDetected:
                    hash = error.computeHash()

                    if hash not in errorHashes:
                        errorHashes.add(hash)
                        uniqueErrors.append(error)

            if stepsRemaining % agentConfig['testing_reset_agent_period'] == (agentConfig['testing_reset_agent_period'] - 1):
                subProcessCommandQueue.put("quit")
                subProcess.terminate()

                subProcessCommandQueue = multiprocessing.Queue()
                subProcessResultQueue = multiprocessing.Queue()
                subProcess = multiprocessing.Process(target=predictedActionSubProcess, args=(shouldBeRandom, environment.branchFeatureSize(), subProcessCommandQueue, subProcessResultQueue))
                subProcess.start()

            del images, envActionMaps, branchFeature, decayingExecutionTraceFeature, additionalFeatures, traces
            print("", end="", sep="", flush=True)

        subProcessCommandQueue.put("quit")
        subProcess.terminate()

        videoPaths = environment.createMovies()

        kwolaVideoDirectory = config.getKwolaUserDataDirectory("videos")
        kwolaDebugVideoDirectory = config.getKwolaUserDataDirectory("debug_videos")

        for sessionN, videoPath, executionSession in zip(range(len(videoPaths)), videoPaths, executionSessions):
            with open(videoPath, 'rb') as origFile:
                with open(os.path.join(kwolaVideoDirectory, f'{str(executionSession.id)}.mp4'), "wb") as cloneFile:
                    cloneFile.write(origFile.read())

        for session in executionSessions:
            session.save()

        testSequence.bugsFound = len(uniqueErrors)
        testSequence.errors = uniqueErrors

        testSequence.status = "completed"

        testSequence.endTime = datetime.now()

        testSequence.executionSessions = executionSessions
        testSequence.save()

        if not shouldBeRandom:
            agent = DeepLearningAgent(config.getAgentConfiguration(), whichGpu=None)
            agent.initialize(environment.branchFeatureSize())
            agent.load()

            videoData = agent.createDebugVideoForExecutionSession(executionSessions[0])
            with open(os.path.join(kwolaDebugVideoDirectory, f'{str(executionSessions[0].id)}.mp4'), "wb") as cloneFile:
                cloneFile.write(videoData)

            del agent

        environment.shutdown()
    except Exception as e:
        traceback.print_exc()
        print("Unhandled exception occurred during testing sequence", flush=True)

    del environment

    # This print statement will trigger the parent manager process to kill this process.
    print("==== Finished Running Testing Sequence! ====", flush=True)

    return returnValue

@app.task
def runTestingSequenceTask(testingSequenceId, shouldBeRandom=False):
    runTestingSequence(testingSequenceId, shouldBeRandom)


if __name__ == "__main__":
    task = TaskProcess(runTestingSequence)
    task.run()

