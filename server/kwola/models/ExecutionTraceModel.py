from mongoengine import *
import datetime
from .actions.BaseAction import BaseAction
from .errors.BaseError import BaseError

class ExecutionTrace(Document):
    time = DateField()

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

    # This field is compressed using a transparent algorithm that makes sparse arrays smaller
    startCumulativeBranchExecutionTraceCompressed = ListField()

    # This field is compressed using a transparent algorithm that makes sparse arrays smaller
    startDecayingExecutionTraceCompressed = ListField()

    # This field is compressed using a transparent algorithm that makes sparse arrays smaller
    branchExecutionTraceCompressed = ListField()

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
            if type(value) == "int" or type(value) == "float":
                newArray.extend([0] * int(value))
            elif isinstance(value, list):
                newArray.extend(value)
            else:
                print(f"ExecutionTrace.decompressArray Error! Unexpected value of type {type(value)} while decompressing array.")


        return newArray
