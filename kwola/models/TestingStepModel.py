from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .ExecutionSessionModel import ExecutionSession
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
from .utilities import saveObjectToDisk, loadObjectFromDisk

class TestingStep(Document):
    id = CustomIDField()

    version = StringField(max_length=200, required=False)

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    bugsFound = IntField(max_length=200, required=False)

    status = StringField(default="fresh")

    executionSessions = ListField(StringField())

    errors = EmbeddedDocumentListField(BaseError)


    def saveToDisk(self, config):
        saveObjectToDisk(self, "testing_steps", config)

    @staticmethod
    def loadFromDisk(id, config):
        data = loadObjectFromDisk(TestingStep, id, "testing_steps", config)

        if data.startTime is not None:
            data.startTime = datetime.datetime(year=data.startTime.year, month=data.startTime.month, day=data.startTime.day, hour=data.startTime.hour, minute=data.startTime.minute, second=data.startTime.second)
        if data.endTime is not None:
            data.endTime = datetime.datetime(year=data.endTime.year, month=data.endTime.month, day=data.endTime.day, hour=data.endTime.hour, minute=data.endTime.minute, second=data.endTime.second)

        return data