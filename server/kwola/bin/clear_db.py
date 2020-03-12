from kwola.models.ApplicationModel import ApplicationModel
from kwola.models.TrainingSequenceModel import TrainingSequence
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.models.ExecutionTraceModel import ExecutionTrace
from mongoengine import connect

connect('kwola')

def main():
    ApplicationModel.objects().delete()
    TrainingSequence.objects().delete()
    TestingSequenceModel.objects().delete()
    ExecutionSession.objects().delete()
    ExecutionTrace.objects().delete()

    print("Kwola Mongo database is now cleared")
