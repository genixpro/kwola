from mongoengine import *
import os.path
from kwola.models.id import CustomIDField
from .utilities import saveObjectToDisk, loadObjectFromDisk



class TrainingSequence(Document):
    id = CustomIDField()

    applicationId = StringField()

    status = StringField()

    startTime = DateTimeField()

    endTime = DateTimeField()

    trainingStepsCompleted = IntField()

    initializationTestingSteps = ListField(StringField())

    testingSteps = ListField(StringField())

    trainingSteps = ListField(StringField())

    averageTimePerStep = FloatField()


    def saveToDisk(self, config):
        saveObjectToDisk(self, "training_sequences", config)


    @staticmethod
    def loadFromDisk(id, config):
        return loadObjectFromDisk(TrainingSequence, id, "training_sequences", config)


