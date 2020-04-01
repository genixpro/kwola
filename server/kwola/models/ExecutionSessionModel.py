from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
import json
import gzip
from .lockedfile import LockedFile

class ExecutionSession(Document):
    id = CustomIDField()

    testingStepId = StringField()

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    executionTraces = ListField(StringField())

    tabNumber = IntField()

    totalReward = FloatField()

    def saveToDisk(self, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("execution_sessions"), str(self.id) + ".json.gz")
        with LockedFile(fileName, 'wb') as f:
            f.write(gzip.compress(bytes(json.dumps(json.loads(self.to_json()), indent=4), "utf8")))


    @staticmethod
    def loadFromDisk(id, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("execution_sessions"), str(id) + ".json.gz")
        if not os.path.exists(fileName):
            return None
        with LockedFile(fileName, 'rb') as f:
            return ExecutionSession.from_json(str(gzip.decompress(f.read()), "utf8"))

