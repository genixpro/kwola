import kwola.tasks.RunTestingSequence
from kwola.models.TestingSequenceModel import TestingSequenceModel
import mongoengine



def main():
    mongoengine.connect('kwola')
    testingSequence = TestingSequenceModel()
    testingSequence.save()

    kwola.tasks.RunTestingSequence.runTestingSequence(str(testingSequence.id))


