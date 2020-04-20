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


from .ActionMapModel import ActionMap
from .actions.BaseAction import BaseAction
from .errors.BaseError import BaseError
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *
import numpy

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

    # This field is stores the execution trace for this round.
    # The dictionary maps file names to lists which then contain the line execution counts for that file
    # It is transparently compressed and decompressed on the fly
    branchTraceCompressed = DictField(ListField())

    # This field is used by the training routine to track how much loss the network had on this execution trace.
    lastTrainingRewardLoss = FloatField(default=1.0)

    # Cached cumulative branch trace vector at the start of this trace, e.g. before the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedStartCumulativeBranchTrace = DictField(ListField())

    # Cached decaying branch trace vector at the start of this trace, e.g. before the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedStartDecayingBranchTrace = DictField(ListField())

    # Cached cumulative branch trace vector at the end of this trace, e.g. after the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedEndCumulativeBranchTrace = DictField(ListField())

    # Cached decaying branch trace vector at the end of this trace, e.g. after the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedEndDecayingBranchTrace = DictField(ListField())

    # Cached decaying branch trace vector at the start of this trace, e.g. before the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    # To be clear, this is a 'future' branch trace, so at the start of the trace,
    # the future includes the actions being performed in this frame.
    cachedStartDecayingFutureBranchTrace = DictField(ListField())

    # Cached decaying branch trace vector at the end of this trace, e.g. after the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedEndDecayingFutureBranchTrace = DictField(ListField())

    # We use Python getter / setter methods to transparently compress and decompress
    # these fields as they go into and out of the database model.
    @property
    def branchTrace(self):
        return {
            fileName: self.decompressArray(self.branchTraceCompressed[fileName])
            for fileName in self.branchTraceCompressed.keys()
        }

    @branchTrace.setter
    def branchTrace(self, value):
        self.branchTraceCompressed = {
            fileName: self.compressArray(value[fileName])
            for fileName in value.keys()
        }

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
        self.cachedStartCumulativeBranchTrace = None
        self.cachedStartDecayingBranchTrace = None
        self.cachedEndCumulativeBranchTrace = None
        self.cachedEndDecayingBranchTrace = None
        self.cachedDecayingFutureBranchTrace = None
        saveObjectToDisk(self, "execution_traces", config)


    @staticmethod
    def loadFromDisk(id, config, omitLargeFields=False, printErrorOnFailure=True):
        trace = loadObjectFromDisk(ExecutionTrace, id, "execution_traces", config, printErrorOnFailure=printErrorOnFailure)
        if trace is not None:
            if omitLargeFields:
                trace.branchExecutionTrace = []
                trace.startDecayingExecutionTrace = []
                trace.startCumulativeBranchExecutionTrace = []

        return trace
