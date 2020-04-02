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
        return loadObjectFromDisk(TestingStep, id, "testing_steps", config)

