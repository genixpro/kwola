from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError
import os.path
from kwola.models.id import CustomIDField
from .utilities import saveObjectToDisk, loadObjectFromDisk


class BugModel(Document):
    id = CustomIDField()

    applicationId = StringField()

    testingStepId = StringField()

    error = EmbeddedDocumentField(BaseError)

    reproductionTraces = ListField(StringField())

    def saveToDisk(self, config):
        saveObjectToDisk(self, "bugs", config)


    @staticmethod
    def loadFromDisk(id, config):
        return loadObjectFromDisk(BugModel, id, "bugs", config)

