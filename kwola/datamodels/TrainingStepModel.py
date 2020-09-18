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

class TrainingStep(Document):
    meta = {
        'indexes': [
            ('owner',),
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    trainingSequenceId = StringField()

    testingRunId = StringField()

    status = StringField()

    hadNaN = BooleanField()

    numberOfIterationsCompleted = IntField()

    averageLoss = FloatField()

    startTime = DateTimeField()

    endTime = DateTimeField()

    averageTimePerIteration = FloatField()

    presentRewardLosses = ListField(FloatField())

    discountedFutureRewardLosses = ListField(FloatField())

    tracePredictionLosses = ListField(FloatField())

    executionFeaturesLosses = ListField(FloatField())

    predictedCursorLosses = ListField(FloatField())

    totalRewardLosses = ListField(FloatField())

    totalLosses = ListField(FloatField())

    totalRebalancedLosses = ListField(FloatField())

    stateValueLosses = ListField(FloatField())

    advantageLosses = ListField(FloatField())

    actionProbabilityLosses = ListField(FloatField())


    def saveToDisk(self, config):
        saveObjectToDisk(self, "training_steps", config)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(TrainingStep, id, "training_steps", config, printErrorOnFailure=printErrorOnFailure)

