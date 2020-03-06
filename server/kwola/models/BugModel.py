from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError



class BugModel(Document):
    applicationId = StringField()

    testingSequenceId = StringField()

    error = EmbeddedDocumentField(BaseError)

    reproductionTraces = ListField(GenericReferenceField(ExecutionTrace))




