from kwola.models.ApplicationModel import ApplicationModel
from kwola.models.TrainingSequenceModel import TrainingSequence
from kwola.models.TestingStepModel import TestingStep
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.models.ExecutionTraceModel import ExecutionTrace
from kwola.models.TrainingStepModel import TrainingStep
from mongoengine import connect


def main():
    ApplicationModel.objects().delete()
    TrainingSequence.objects().delete()
    TestingStep.objects().delete()
    ExecutionSession.objects().delete()
    ExecutionTrace.objects().delete()
    TrainingStep.objects().delete()

    print("Kwola Mongo database is now cleared")
