from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from .RunTrainingStep import runTrainingStep
from .RunTestingSequence import runTestingSequence




@app.task
def trainAgent():
    sequencesNeeded = 1000

    sequencesCompleted = 0

    while sequencesCompleted < sequencesNeeded:
        sequencesCompleted += 1

        sequence = TestingSequenceModel()
        sequence.save()

        runTestingSequence(str(sequence.id))

        print("Testing Sequence Completed")

        runTrainingStep()



if __name__ == "__main__":
    trainAgent()

