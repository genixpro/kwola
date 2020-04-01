from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .ExecutionSessionModel import ExecutionSession
from .errors.BaseError import BaseError
from kwola.config.config import getKwolaUserDataDirectory
import os.path
from kwola.models.id import CustomIDField
import json

class TestingStep(Document):
    id = CustomIDField()

    version = StringField(max_length=200, required=False)

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    bugsFound = IntField(max_length=200, required=False)

    status = StringField(default="fresh")

    executionSessions = ListField(StringField())

    errors = EmbeddedDocumentListField(BaseError)


    def saveToDisk(self):
        fileName = os.path.join(getKwolaUserDataDirectory("testing_steps"), str(self.id) + ".json")
        with open(fileName, 'wt') as f:
            f.write(json.dumps(json.loads(self.to_json()), indent=4))


    @staticmethod
    def loadFromDisk(id):
        fileName = os.path.join(getKwolaUserDataDirectory("testing_steps"), str(id) + ".json")
        if not os.path.exists(fileName):
            return None
        with open(fileName, 'rt') as f:
            return TestingStep.from_json(f.read())
