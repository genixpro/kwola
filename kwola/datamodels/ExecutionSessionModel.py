#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *

class ExecutionSession(Document):
    meta = {
        'indexes': [
            ('owner',),
            ('owner', 'testingStepId', 'startTime'),
            ('owner', 'testingRunId', 'startTime'),
            ('testingRunId', 'startTime')
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    testingStepId = StringField()

    status = StringField(default="completed")

    testingRunId = StringField(required=False)

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    executionTraces = ListField(StringField())

    tabNumber = IntField()

    totalReward = FloatField()

    browser = StringField()

    userAgent = StringField()

    windowSize = StringField()

    useForFutureChangeDetection = BooleanField(default=False)

    isChangeDetectionSession = BooleanField(default=False)

    changeDetectionPriorExecutionSessionId = StringField()

    executionTracesWithChanges = ListField(StringField())

    bestApplicationProvidedCumulativeFitness = FloatField()

    countTracesWithNewBranches = FloatField(default=None)

    def saveToDisk(self, config):
        saveObjectToDisk(self, "execution_sessions", config)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(ExecutionSession, id, "execution_sessions", config, printErrorOnFailure=printErrorOnFailure)

