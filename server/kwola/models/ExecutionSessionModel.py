from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
from .utilities import saveObjectToDisk, loadObjectFromDisk

class ExecutionSession(Document):
    id = CustomIDField()

    testingStepId = StringField()

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    executionTraces = ListField(StringField())

    tabNumber = IntField()

    totalReward = FloatField()

    def saveToDisk(self, config):
        saveObjectToDisk(self, "execution_sessions", config)


    @staticmethod
    def loadFromDisk(id, config):
        return loadObjectFromDisk(ExecutionSession, id, "execution_sessions", config)

