from .BaseError import BaseError
from mongoengine import *
import hashlib


class ExceptionError(BaseError):
    """
        This class represents the most typical scenario of error - when an uncaught exception bubbles to the top.
    """

    stacktrace = StringField()

    message = StringField()

    def computeHash(self):
        hasher = hashlib.md5()
        hasher.update(self.stacktrace)

        return hasher.hexdigest()



