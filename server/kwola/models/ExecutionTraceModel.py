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

    networkTrafficTrace = ListField(IntField())

    dataChangeTrace = ListField(IntField())

    beforeScreenshot = StringField()

    afterScreenshot = StringField()

    startURL = StringField()

    finishURL = StringField()

    didErrorOccur = BooleanField()

    didCodeExecute = BooleanField()

    didNewBranchesExecute = BooleanField()

    didDataChange = BooleanField()

    didNewDataFieldsChange = BooleanField()

    didScreenshotChange = BooleanField()

    didURLChange = BooleanField()

    cumulativeBranchCoverage = FloatField()

