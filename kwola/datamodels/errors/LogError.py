#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .BaseError import BaseError
from mongoengine import *
import hashlib


class LogError(BaseError):
    """
        This class represents when something is logged at a log-level that is considered an error. The exact details are user definable but effectively this is an error
        that is only detected in the log files.
    """

    logLevel = StringField()



    def computeHash(self):
        hasher = hashlib.sha256()
        hasher.update(bytes(self.message, "utf8"))
        hasher.update(bytes(self.logLevel, "utf8"))

        return hasher.hexdigest()

    def generateErrorDescription(self):
        return self.message