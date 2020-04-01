from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .ExecutionSessionModel import ExecutionSession
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
import json
import gzip
from .lockedfile import LockedFile

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
        fileName = os.path.join(config.getKwolaUserDataDirectory("testing_steps"), str(self.id) + ".json.gz")
        with LockedFile(fileName, 'wb') as f:
            f.write(gzip.compress(bytes(json.dumps(json.loads(self.to_json()), indent=4), "utf8")))

    @staticmethod
    def loadFromDisk(id, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("testing_steps"), str(id) + ".json.gz")
        if not os.path.exists(fileName):
            return None
        with LockedFile(fileName, 'rb') as f:
            return TestingStep.from_json(str(gzip.decompress(f.read()), "utf8"))
