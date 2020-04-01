from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
from kwola.config.config import getKwolaUserDataDirectory
import os.path
from kwola.models.id import CustomIDField
import json



class BugModel(Document):
    id = CustomIDField()

    applicationId = StringField()

    testingStepId = StringField()

    error = EmbeddedDocumentField(BaseError)

    reproductionTraces = ListField(StringField())

    def saveToDisk(self):
        fileName = os.path.join(getKwolaUserDataDirectory("bugs"), str(self.id) + ".json")
        with open(fileName, 'wt') as f:
            f.write(json.dumps(json.loads(self.to_json()), indent=4))


    @staticmethod
    def loadFromDisk(id):
        fileName = os.path.join(getKwolaUserDataDirectory("bugs"), str(id) + ".json")
        with open(fileName, 'rt') as f:
            return BugModel.from_json(f.read())
