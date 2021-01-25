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


from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *

class ExecutionSession(Document):
    meta = {
        'indexes': [
            ('owner',),
            ('owner', 'testingStepId', 'startTime'),
            ('owner', 'testingRunId', 'startTime'),
            ('testingRunId', 'startTime')
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    testingStepId = StringField()

    status = StringField(default="completed")

    testingRunId = StringField(required=False)

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    executionTraces = ListField(StringField())

    tabNumber = IntField()

    totalReward = FloatField()

    browser = StringField()

    userAgent = StringField()

    windowSize = StringField()

    useForFutureChangeDetection = BooleanField(default=False)

    isChangeDetectionSession = BooleanField(default=False)

    changeDetectionPriorExecutionSessionId = StringField()

    executionTracesWithChanges = ListField(StringField())

    def saveToDisk(self, config):
        saveObjectToDisk(self, "execution_sessions", config)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(ExecutionSession, id, "execution_sessions", config, printErrorOnFailure=printErrorOnFailure)

