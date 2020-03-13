from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.components.agents.RandomAgent import RandomAgent
from kwola.components.TaskProcess import TaskProcess
import random
from datetime import datetime
import traceback
from bson import ObjectId
import os
from kwola.config import config

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

        if shouldBeRandom:
            agent = RandomAgent()
        else:
            agent = DeepLearningAgent(config.getAgentConfiguration(), whichGpu=None)

        agent.initialize(environment)
        agent.load()

        step = 0

        while stepsRemaining > 0:
            stepsRemaining -= 1

            actions = agent.nextBestActions(step)
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
            print("", end="", sep="", flush=True)


        videoPaths = environment.createMovies()

        kwolaVideoDirectory = config.getKwolaUserDataDirectory("videos")
        kwolaDebugVideoDirectory = config.getKwolaUserDataDirectory("debug_videos")

        for sessionN, videoPath, executionSession in zip(range(len(videoPaths)), videoPaths, executionSessions):
            with open(videoPath, 'rb') as origFile:
                with open(os.path.join(kwolaVideoDirectory, f'{str(executionSession.id)}.mp4'), "wb") as cloneFile:
                    cloneFile.write(origFile.read())

            if not shouldBeRandom and sessionN == 0:
                videoData = agent.createDebugVideoForExecutionSession(executionSession)
                with open(os.path.join(kwolaDebugVideoDirectory, f'{str(executionSession.id)}.mp4'), "wb") as cloneFile:
                    cloneFile.write(videoData)

        for session in executionSessions:
            session.save()

        testSequence.bugsFound = len(uniqueErrors)
        testSequence.errors = uniqueErrors

        testSequence.status = "completed"

        testSequence.endTime = datetime.now()

        testSequence.executionSessions = executionSessions
        testSequence.save()

        environment.shutdown()
    except Exception as e:
        traceback.print_exc()
        print("Unhandled exception occurred during testing sequence", flush=True)

    del environment, agent

    # This print statement will trigger the parent manager process to kill this process.
    print("==== Finished Running Testing Sequence! ====", flush=True)

    return returnValue

@app.task
def runTestingSequenceTask(testingSequenceId, shouldBeRandom=False):
    runTestingSequence(testingSequenceId, shouldBeRandom)


if __name__ == "__main__":
    task = TaskProcess(runTestingSequence)
    task.run()

