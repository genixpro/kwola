from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError



class TrainingStep(Document):
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





