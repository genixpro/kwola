#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .ActionMapModel import ActionMap
from .actions.BaseAction import BaseAction
from .errors.BaseError import BaseError
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *
import numpy
import scipy.sparse

class ExecutionTrace(Document):
    meta = {
        'indexes': [
            ('owner',),
            ('owner', 'executionSessionId'),
            ('owner', 'testingStepId'),
            ('executionSessionId',)
        ]
    }

    id = CustomIDField()

    owner = StringField()

    time = DateField()

    applicationId = StringField()

    executionSessionId = StringField()

    testingStepId = StringField()

    testingRunId = StringField()

    actionMaps = EmbeddedDocumentListField(ActionMap)

    actionPerformed = EmbeddedDocumentField(BaseAction)

    errorsDetected = EmbeddedDocumentListField(BaseError)

    logOutput = StringField()

    networkTrafficTrace = ListField(StringField())

    startScreenshotHash = StringField()

    finishScreenshotHash = StringField()

    frameNumber = IntField()

    traceNumber = IntField()

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

    browser = StringField()

    userAgent = StringField()

    windowSize = StringField()

    codePrevalenceScore = FloatField()

    codePrevalenceLogNormalizedZScore = FloatField()

    # This field is stores the execution trace for this round.
    # The dictionary maps file names to lists which then contain the line execution counts for that file
    # It is transparently compressed and decompressed on the fly
    branchTraceCompressed = DictField(ListField())

    # This field is used by the training routine to track how much loss the network had on this execution trace.
    lastTrainingRewardLoss = FloatField(default=1.0)

    # This is a cache of the uncompressed branch trace
    cachedUncompressedBranchTrace = DictField(DynamicField(), default=None)

    # Cached cumulative branch trace vector at the start of this trace, e.g. before the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedStartCumulativeBranchTrace = DictField(DynamicField(), default=None)

    # Cached decaying branch trace vector at the start of this trace, e.g. before the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedStartDecayingBranchTrace = DictField(DynamicField(), default=None)

    # Cached cumulative branch trace vector at the end of this trace, e.g. after the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedEndCumulativeBranchTrace = DictField(DynamicField(), default=None)

    # Cached decaying branch trace vector at the end of this trace, e.g. after the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedEndDecayingBranchTrace = DictField(DynamicField(), default=None)

    # Cached decaying branch trace vector at the start of this trace, e.g. before the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    # To be clear, this is a 'future' branch trace, so at the start of the trace,
    # the future includes the actions being performed in this frame.
    cachedStartDecayingFutureBranchTrace = DictField(DynamicField(), default=None)

    # Cached decaying branch trace vector at the end of this trace, e.g. after the action was ran.
    # This is only "cached" because it can actually be recomputed on the fly
    cachedEndDecayingFutureBranchTrace = DictField(DynamicField(), default=None)

    # The cached recent actions image is an image that shows exactly where specific actions were performed
    # on the screen
    cachedStartingRecentActionsImage = DynamicField(default=None)

    # The cached recent actions image is an image that shows exactly where specific actions were performed
    # on the screen
    cachedEndingRecentActionsImage = DynamicField(default=None)

    timeForScreenshot = FloatField()
    timeForActionMapRetrieval = FloatField()
    timeForActionDecision = FloatField()
    timeForActionExecution = FloatField()
    timeForMiscellaneous = FloatField()

    actionExecutionTimes = DictField(FloatField())

    # This records the application provided cumulative fitness value, indicating how well the model is doing on
    # on this particular execution trace. This is often used in Kwola's internal experiments, which have their own
    # internal measurements of the how good kwola's result was. This value is recorded prior to the action for this
    # executions trace
    startApplicationProvidedCumulativeFitness = FloatField()

    # This records the application provided cumulative fitness value, indicating how well the model is doing on
    # on this particular execution trace. This is often used in Kwola's internal experiments, which have their own
    # internal measurements of the how good kwola's result was. This value is recorded after this execution trace.
    endApplicationProvidedCumulativeFitness = FloatField()

    # We use Python getter / setter methods to transparently compress and decompress
    # these fields as they go into and out of the database model.
    @property
    def branchTrace(self):
        if self.cachedUncompressedBranchTrace is None:
            self.cachedUncompressedBranchTrace = {
                fileName: self.decompressArray(self.branchTraceCompressed[fileName])
                for fileName in self.branchTraceCompressed.keys()
            }

        return self.cachedUncompressedBranchTrace

    @branchTrace.setter
    def branchTrace(self, value):
        self.branchTraceCompressed = {
            fileName: self.compressArray(value[fileName])
            for fileName in value.keys()
        }

        self.cachedUncompressedBranchTrace = {
            fileName: self.decompressArray(self.branchTraceCompressed[fileName])
            for fileName in self.branchTraceCompressed.keys()
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

        newArray = numpy.array(newArray)

        return scipy.sparse.csc_matrix(numpy.reshape(newArray, newshape=[newArray.shape[0], 1]), shape=[newArray.shape[0], 1], dtype=numpy.float64)

    def saveToDisk(self, config):
        cachedUncompressedBranchTrace = self.cachedUncompressedBranchTrace
        cachedStartCumulativeBranchTrace = self.cachedStartCumulativeBranchTrace
        cachedStartDecayingBranchTrace = self.cachedStartDecayingBranchTrace
        cachedEndCumulativeBranchTrace = self.cachedEndCumulativeBranchTrace
        cachedEndDecayingBranchTrace = self.cachedEndDecayingBranchTrace
        cachedStartDecayingFutureBranchTrace = self.cachedStartDecayingFutureBranchTrace
        cachedEndDecayingFutureBranchTrace = self.cachedEndDecayingFutureBranchTrace
        cachedStartingRecentActionsImage = self.cachedStartingRecentActionsImage
        cachedEndingRecentActionsImage = self.cachedEndingRecentActionsImage

        self.cachedUncompressedBranchTrace = None
        self.cachedStartCumulativeBranchTrace = None
        self.cachedStartDecayingBranchTrace = None
        self.cachedEndCumulativeBranchTrace = None
        self.cachedEndDecayingBranchTrace = None
        self.cachedStartDecayingFutureBranchTrace = None
        self.cachedEndDecayingFutureBranchTrace = None
        self.cachedStartingRecentActionsImage = None
        self.cachedEndingRecentActionsImage = None
        saveObjectToDisk(self, "execution_traces", config)
        self.cachedUncompressedBranchTrace = cachedUncompressedBranchTrace
        self.cachedStartCumulativeBranchTrace = cachedStartCumulativeBranchTrace
        self.cachedStartDecayingBranchTrace = cachedStartDecayingBranchTrace
        self.cachedEndCumulativeBranchTrace = cachedEndCumulativeBranchTrace
        self.cachedEndDecayingBranchTrace = cachedEndDecayingBranchTrace
        self.cachedStartDecayingFutureBranchTrace = cachedStartDecayingFutureBranchTrace
        self.cachedEndDecayingFutureBranchTrace = cachedEndDecayingFutureBranchTrace
        self.cachedStartingRecentActionsImage = cachedStartingRecentActionsImage
        self.cachedEndingRecentActionsImage = cachedEndingRecentActionsImage


    @staticmethod
    def loadFromDisk(id, config, omitLargeFields=False, printErrorOnFailure=True, applicationId=None):
        trace = loadObjectFromDisk(ExecutionTrace, id, "execution_traces", config, printErrorOnFailure=printErrorOnFailure, applicationId=applicationId)
        if trace is not None:
            if omitLargeFields:
                trace.branchExecutionTrace = []
                trace.actionMaps = []

        return trace
