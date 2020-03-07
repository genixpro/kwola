from mongoengine import *
import datetime
from .actions.BaseAction import BaseAction
from .errors.BaseError import BaseError

class ExecutionTrace(Document):
    time = DateField()

    actionPerformed = EmbeddedDocumentField(BaseAction)

    errorsDetected = EmbeddedDocumentListField(BaseError)

    logOutput = StringField()

    branchExecutionTrace = ListField(IntField())

    startCumulativeBranchExecutionTrace = ListField(IntField())

    networkTrafficTrace = ListField(StringField())

    startScreenshotHash = StringField()

    finishScreenshotHash = StringField()

    frameNumber = IntField()

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
