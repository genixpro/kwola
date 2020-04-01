from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
import json
import gzip
from .lockedfile import LockedFile


class BugModel(Document):
    id = CustomIDField()

    applicationId = StringField()

    testingStepId = StringField()

    error = EmbeddedDocumentField(BaseError)

    reproductionTraces = ListField(StringField())

    def saveToDisk(self, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("bugs"), str(self.id) + ".json.gz")
        with LockedFile(fileName, 'wb') as f:
            f.write(gzip.compress(bytes(json.dumps(json.loads(self.to_json()), indent=4), "utf8")))


    @staticmethod
    def loadFromDisk(id, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("bugs"), str(id) + ".json.gz")
        with LockedFile(fileName, 'rb') as f:
            return BugModel.from_json(str(gzip.decompress(f.read()), "utf8"))

