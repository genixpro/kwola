#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *



class TrainingSequence(Document):
    meta = {
        'indexes': [
            ('owner',),
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    status = StringField()

    startTime = DateTimeField()

    endTime = DateTimeField()

    # Deprecated. Eliminating this variable, replaced with trainingLoopsCompleted.
    trainingStepsCompleted = IntField()
    trainingLoopsCompleted = IntField()

    initializationTestingSteps = ListField(StringField())

    testingSteps = ListField(StringField())

    trainingSteps = ListField(StringField())

    averageTimePerStep = FloatField()

    trainingStepsLaunched = IntField()
    testingStepsLaunched = IntField()


    def saveToDisk(self, config):
        saveObjectToDisk(self, "training_sequences", config)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(TrainingSequence, id, "training_sequences", config, printErrorOnFailure=printErrorOnFailure)


