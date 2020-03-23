import kwola.tasks.RunTestingSequence
from kwola.models.TestingSequenceModel import TestingSequenceModel




def main():
    testingSequence = TestingSequenceModel()
    testingSequence.save()

    kwola.tasks.RunTestingSequence.runTestingSequence(str(testingSequence.id))


