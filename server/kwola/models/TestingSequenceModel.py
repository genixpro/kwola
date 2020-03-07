from mongoengine import *
import datetime
from .ExecutionTraceModel import ExecutionTrace
from .errors.BaseError import BaseError

class TestingSequenceModel(Document):

    version = StringField(max_length=200, required=False)

    startTime = DateField(max_length=200, required=False)
    endTime = DateField(max_length=200, required=False)

    bugsFound = IntField(max_length=200, required=False)

    status = StringField(default="fresh")

    executionTraces = ListField(GenericReferenceField())

    errors = EmbeddedDocumentListField(BaseError)

