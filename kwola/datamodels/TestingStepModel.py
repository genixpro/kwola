#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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