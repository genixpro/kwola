from .BaseError import BaseError
from mongoengine import *
import hashlib


class LogError(BaseError):
    """
        This class represents when something is logged at a log-level that is considered an error. The exact details are user definable but effectively this is an error
        that is only detected in the log files.
    """

    message = StringField()

    logLevel = StringField()



    def computeHash(self):
        hasher = hashlib.md5()
        hasher.update(self.message)

        return hasher.hexdigest()
