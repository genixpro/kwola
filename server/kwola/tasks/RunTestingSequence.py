from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.components.agents.RandomAgent import RandomAgent
import random
from datetime import datetime
from bson import ObjectId
import os

@app.task
def runTestingSequence(testingSequenceId, shouldBeRandom=False):
    environment = WebEnvironment()

    stepsRemaining = 100

    testSequence = TestingSequenceModel.objects(id=testingSequenceId).first()

    testSequence.startTime = datetime.now()
    testSequence.status = "running"
    testSequence.save()

    executionTraces = []

    errorHashes = set()
    uniqueErrors = []

    if shouldBeRandom:
        agent = RandomAgent()
    else:
        agent = DeepLearningAgent()

    agent.initialize(environment)
    agent.load()

    while stepsRemaining > 0:
        stepsRemaining -= 1

        action = agent.nextBestAction()

        trace = environment.runAction(action)
        trace.save()

        executionTraces.append(trace)

        for error in trace.errorsDetected:
            hash = error.computeHash()

            if hash not in errorHashes:
                errorHashes.add(hash)
                uniqueErrors.append(error)

    videoPath = environment.createMovie()


    with open(videoPath, 'rb') as origFile:
        with open(f'/home/bradley/{str(testSequence.id)}.mp4', "wb") as cloneFile:
            cloneFile.write(origFile.read())

    testSequence.bugsFound = len(uniqueErrors)
    testSequence.errors = uniqueErrors

    testSequence.status = "completed"

    testSequence.endTime = datetime.now()
    testSequence.executionTraces = executionTraces
    testSequence.save()

    environment.shutdown()

    print("Finished Running Testing Sequence! Yay!")

    return ""


