from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
from kwola.config.config import getKwolaUserDataDirectory
import os.path
from kwola.models.id import CustomIDField
import json
import gzip

class ExecutionSession(Document):
    id = CustomIDField()

    testingStepId = StringField()

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    executionTraces = ListField(StringField())

    tabNumber = IntField()

    totalReward = FloatField()

    def saveToDisk(self):
        fileName = os.path.join(getKwolaUserDataDirectory("execution_sessions"), str(self.id) + ".json.gz")
        with gzip.open(fileName, 'wt') as f:
            f.write(json.dumps(json.loads(self.to_json()), indent=4))


    @staticmethod
    def loadFromDisk(id):
        fileName = os.path.join(getKwolaUserDataDirectory("execution_sessions"), str(id) + ".json.gz")
        if not os.path.exists(fileName):
            return None
        with gzip.open(fileName, 'rt') as f:
            return ExecutionSession.from_json(f.read())
