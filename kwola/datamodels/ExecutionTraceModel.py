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


from mongoengine import *
from datetime import datetime
from .actions.BaseAction import BaseAction
from .errors.BaseError import BaseError
from .ActionMapModel import ActionMap
import numpy
import os.path
from .id import CustomIDField
from .utilities import saveObjectToDisk, loadObjectFromDisk

class ExecutionTrace(Document):
    id = CustomIDField()

    time = DateField()

    executionSessionId = StringField()

    testingStepId = StringField()

    actionMaps = EmbeddedDocumentListField(ActionMap)

    actionPerformed = EmbeddedDocumentField(BaseAction)

    errorsDetected = EmbeddedDocumentListField(BaseError)

    logOutput = StringField()

    networkTrafficTrace = ListField(StringField())

    startScreenshotHash = StringField()

    finishScreenshotHash = StringField()

    frameNumber = IntField()

    tabNumber = IntField()

    startURL = StringField()

    finishURL = StringField()

    didActionSucceed = BooleanField()

    didErrorOccur = BooleanField()

    didNewErrorOccur = BooleanField()

    didCodeExecute = BooleanField()

    didNewBranchesExecute = BooleanField()

    hadNetworkTraffic = BooleanField()

    hadNewNetworkTraffic = BooleanField()

    didScreenshotChange = BooleanField()

    isScreenshotNew = BooleanField()

    didURLChange = BooleanField()

    isURLNew = BooleanField()

    hadLogOutput = BooleanField()

    cumulativeBranchCoverage = FloatField()

    cursor = StringField()

    # This field is compressed using a transparent algorithm that makes sparse arrays smaller
    startCumulativeBranchExecutionTraceCompressed = ListField()

    # This field is compressed using a transparent algorithm that makes sparse arrays smaller
    startDecayingExecutionTraceCompressed = ListField()

    # This field is compressed using a transparent algorithm that makes sparse arrays smaller
    branchExecutionTraceCompressed = ListField()

    # This field is used by the training routine to track how much loss the network had on this execution trace.
    lastTrainingRewardLoss = FloatField(default=1.0)

    # We use Python getter / setter methods to transparently compress and decompress
    # these fields as they go into and out of the database model.
    @property
    def startCumulativeBranchExecutionTrace(self):
        return self.decompressArray(self.startCumulativeBranchExecutionTraceCompressed)

    @startCumulativeBranchExecutionTrace.setter
    def startCumulativeBranchExecutionTrace(self, value):
        self.startCumulativeBranchExecutionTraceCompressed = self.compressArray(value)


    @property
    def startDecayingExecutionTrace(self):
        return self.decompressArray(self.startDecayingExecutionTraceCompressed)

    @startDecayingExecutionTrace.setter
    def startDecayingExecutionTrace(self, value):
        self.startDecayingExecutionTraceCompressed = self.compressArray(value)


    @property
    def branchExecutionTrace(self):
        return self.decompressArray(self.branchExecutionTraceCompressed)

    @branchExecutionTrace.setter
    def branchExecutionTrace(self, value):
        self.branchExecutionTraceCompressed = self.compressArray(value)


    def compressArray(self, array):
        """
            Utility function that compresses numpy arrays into a very simple sparse format that is vastly more efficient for storage

            :return:
        """
        newArray = []

        mode = "zero"
        zeroCount = 0
        nonZeroArray = []

        for value in array:
            if value == 0:
                if mode == "zero":
                    zeroCount += 1
                else:
                    newArray.append(nonZeroArray)
                    nonZeroArray = []
                    mode = "zero"
                    zeroCount = 1
            else:
                if mode == "zero":
                    newArray.append(zeroCount)
                    zeroCount = 0
                    mode = "data"
                    nonZeroArray.append(value)
                else:
                    nonZeroArray.append(value)

        if mode == "zero":
            newArray.append(zeroCount)
        else:
            newArray.append(nonZeroArray)

        return newArray


    def decompressArray(self, array):
        """
            Utility function that decompresses numpy arrays which were first compressed by ExecutionTrace.compressArray

            :return:
        """
        newArray = []

        for value in array:
            if isinstance(value, int) or isinstance(value, float):
                newArray.extend([0] * int(value))
            elif isinstance(value, list):
                newArray.extend(value)
            else:
                print(f"ExecutionTrace.decompressArray Error! Unexpected value of type {type(value)} while decompressing array.")


        return numpy.array(newArray)

    def saveToDisk(self, config):
        saveObjectToDisk(self, "execution_traces", config)


    @staticmethod
    def loadFromDisk(id, config, omitLargeFields=False):
        trace = loadObjectFromDisk(ExecutionTrace, id, "execution_traces", config)
        if trace is not None:
            if omitLargeFields:
                trace.branchExecutionTrace = []
                trace.startDecayingExecutionTrace = []
                trace.startCumulativeBranchExecutionTrace = []

        return trace
