from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError

class ExecutionSession(Document):

    testingSequenceId = StringField()

    startTime = DateTimeField(max_length=200, required=False)

    endTime = DateTimeField(max_length=200, required=False)

    executionTraces = ListField(GenericReferenceField())

    tabNumber = IntField()

