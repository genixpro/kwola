#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from mongoengine import *

class ExecutionSessionTraceWeights(Document):
    id = CustomIDField()

    weights = DictField(FloatField())

    def saveToDisk(self, config):
        saveObjectToDisk(self, "execution_session_trace_weights", config)


    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(ExecutionSessionTraceWeights, id, "execution_session_trace_weights", config, printErrorOnFailure=printErrorOnFailure)

