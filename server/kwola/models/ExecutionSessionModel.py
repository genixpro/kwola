from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError

class ExecutionSession(Document):

    startTime = DateField(max_length=200, required=False)

    endTime = DateField(max_length=200, required=False)

    executionTraces = ListField(GenericReferenceField())

    tabNumber = IntField()

