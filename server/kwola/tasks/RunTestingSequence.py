from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.components.agents.RandomAgent import RandomAgent
import random
from datetime import datetime
import traceback
from bson import ObjectId
import os
from kwola.config import config

@app.task
def runTestingSequence(testingSequenceId, shouldBeRandom=False):
    print("Starting New Testing Sequence")
    try:
        environment = WebEnvironment(numberParallelSessions=16)

        stepsRemaining = 100

        testSequence = TestingSequenceModel.objects(id=testingSequenceId).first()

        testSequence.startTime = datetime.now()
        testSequence.status = "running"
        testSequence.save()

        executionSessions = [
            ExecutionSession(startTime=datetime.now(), endTime=None, tabNumber=sessionN, executionTraces=[])
            for sessionN in range(environment.numberParallelSessions())
        ]

        errorHashes = set()
        uniqueErrors = []

        if shouldBeRandom:
            agent = RandomAgent()
        else:
            agent = DeepLearningAgent(whichGpu=None)

        agent.initialize(environment)
        agent.load()

        while stepsRemaining > 0:
            stepsRemaining -= 1

            actions = agent.nextBestActions()

            traces = environment.runActions(actions)
            for sessionN, trace in enumerate(traces):
                trace.save()

                executionSessions[sessionN].executionTraces.append(trace)

                for error in trace.errorsDetected:
                    hash = error.computeHash()

                    if hash not in errorHashes:
                        errorHashes.add(hash)
                        uniqueErrors.append(error)


        videoPaths = environment.createMovies()

        kwolaVideoDirectory = config.getKwolaUserDataDirectory("videos")

        for sessionN, videoPath in enumerate(videoPaths):
            with open(videoPath, 'rb') as origFile:
                with open(os.path.join(kwolaVideoDirectory, f'{str(testSequence.id)}-{sessionN}.mp4'), "wb") as cloneFile:
                    cloneFile.write(origFile.read())

        testSequence.bugsFound = len(uniqueErrors)
        testSequence.errors = uniqueErrors

        testSequence.status = "completed"

        testSequence.endTime = datetime.now()

        for session in executionSessions:
            session.save()

        testSequence.executionSessions = executionSessions
        testSequence.save()

        environment.shutdown()

        print("Finished Running Testing Sequence!")
    except Exception as e:
        traceback.print_exc()
        print("Unhandled exception occurred during testing sequence")

    return ""


