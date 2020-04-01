from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
import json
import gzip
from .lockedfile import LockedFile

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
        fileName = os.path.join(config.getKwolaUserDataDirectory("training_steps"), str(self.id) + ".json.gz")
        with LockedFile(fileName, 'wb') as f:
            f.write(gzip.compress(bytes(json.dumps(json.loads(self.to_json()), indent=4), "utf8")))


    @staticmethod
    def loadFromDisk(id, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("training_steps"), str(id) + ".json.gz")
        if not os.path.exists(fileName):
            return None
        with LockedFile(fileName, 'rb') as f:
            return TrainingStep.from_json(str(gzip.decompress(f.read()), "utf8"))
