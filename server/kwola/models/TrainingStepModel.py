from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
from .utilities import saveObjectToDisk, loadObjectFromDisk

class TrainingStep(Document):
    id = CustomIDField()

    applicationId = StringField()

    trainingSequenceId = StringField()

    status = StringField()

    numberOfIterationsCompleted = IntField()

    averageLoss = FloatField()

    startTime = DateTimeField()

    endTime = DateTimeField()

    averageTimePerIteration = FloatField()

    presentRewardLosses = ListField(FloatField())

    discountedFutureRewardLosses = ListField(FloatField())

    tracePredictionLosses = ListField(FloatField())

    executionFeaturesLosses = ListField(FloatField())

    targetHomogenizationLosses = ListField(FloatField())

    predictedCursorLosses = ListField(FloatField())

    totalRewardLosses = ListField(FloatField())

    totalLosses = ListField(FloatField())

    totalRebalancedLosses = ListField(FloatField())

    stateValueLosses = ListField(FloatField())

    advantageLosses = ListField(FloatField())

    actionProbabilityLosses = ListField(FloatField())


    def saveToDisk(self, config):
        saveObjectToDisk(self, "training_steps", config)


    @staticmethod
    def loadFromDisk(id, config):
        return loadObjectFromDisk(TrainingStep, id, "training_steps", config)

