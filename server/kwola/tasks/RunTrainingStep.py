from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
import random
from datetime import datetime


@app.task
def runTrainingStep():
    environment = WebEnvironment()

    testSequences = list(TestingSequenceModel.objects(status="completed"))

    agent = DeepLearningAgent()
    agent.initialize(environment)
    agent.load()

    random.shuffle(testSequences)

    for sequence in testSequences:
        agent.learnFromTestingSequence(sequence)

    agent.save()

    environment.shutdown()

    return ""


