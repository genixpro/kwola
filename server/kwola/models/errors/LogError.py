from .BaseError import BaseError
from mongoengine import *


class LogError(BaseError):
    """
        This class represents when something is logged at a log-level that is considered an error. The exact details are user definable but effectively this is an error
        that is only detected in the log files.
    """

    message = StringField()

    logLevel = StringField()

