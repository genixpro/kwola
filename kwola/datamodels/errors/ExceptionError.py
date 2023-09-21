#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .BaseError import BaseError
from mongoengine import *
import hashlib


class ExceptionError(BaseError):
    """
        This class represents the most typical scenario of error - when an uncaught exception bubbles to the top.
    """

    stacktrace = StringField()

    source = StringField()

    lineNumber = IntField()

    columnNumber = IntField()

    def computeHash(self):
        hasher = hashlib.sha256()
        hasher.update(bytes(self.stacktrace, "utf8"))

        return hasher.hexdigest()


    def generateErrorDescription(self):
        return self.stacktrace
