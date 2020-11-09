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



class TrainingSequence(Document):
    meta = {
        'indexes': [
            ('owner',),
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    status = StringField()

    startTime = DateTimeField()

    endTime = DateTimeField()

    # Deprecated. Eliminating this variable, replaced with trainingLoopsCompleted.
    trainingStepsCompleted = IntField()
    trainingLoopsCompleted = IntField()

    initializationTestingSteps = ListField(StringField())

    testingSteps = ListField(StringField())

    trainingSteps = ListField(StringField())

    averageTimePerStep = FloatField()

    trainingStepsLaunched = IntField()
    testingStepsLaunched = IntField()


    def saveToDisk(self, config):
        saveObjectToDisk(self, "training_sequences", config)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(TrainingSequence, id, "training_sequences", config, printErrorOnFailure=printErrorOnFailure)


