from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
import random
from datetime import datetime
import traceback


@app.task
def runTrainingStep():
    environment = WebEnvironment()

    testSequences = list(TestingSequenceModel.objects(status="completed"))

    agent = DeepLearningAgent()
    agent.initialize(environment)
    agent.load()

    for repeat in range(10):
        random.shuffle(testSequences)

        for sequence in testSequences[:5]:
            try:
                agent.learnFromTestingSequence(sequence)
            except Exception as e:
                print(f"Error occurred while learning sequence on object {str(sequence.id)}!")
                traceback.print_exc()

    agent.save()

    environment.shutdown()

    return ""


