#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .errors.BaseError import BaseError
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *
import datetime

class TestingStep(Document):
    meta = {
        'indexes': [
            ('owner',)
        ]
    }

    id = CustomIDField()

    owner = StringField()

    version = StringField(required=False)

    startTime = DateTimeField(required=False)

    endTime = DateTimeField(required=False)

    bugsFound = IntField(required=False)

    status = StringField(default="fresh")

    executionSessions = ListField(StringField())

    errors = EmbeddedDocumentListField(BaseError)

    testingRunId = StringField(required=False)

    applicationId = StringField(required=False)

    browser = StringField()

    userAgent = StringField()

    windowSize = StringField()

    testStepIndexWithinRun = IntField()

    def saveToDisk(self, config):
        saveObjectToDisk(self, "testing_steps", config)

    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        data = loadObjectFromDisk(TestingStep, id, "testing_steps", config, printErrorOnFailure=printErrorOnFailure)

        if data is not None:
            if data.startTime is not None:
                data.startTime = datetime.datetime(year=data.startTime.year, month=data.startTime.month, day=data.startTime.day, hour=data.startTime.hour, minute=data.startTime.minute, second=data.startTime.second)
            if data.endTime is not None:
                data.endTime = datetime.datetime(year=data.endTime.year, month=data.endTime.month, day=data.endTime.day, hour=data.endTime.hour, minute=data.endTime.minute, second=data.endTime.second)

        return data