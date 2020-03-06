from .BaseError import BaseError
from mongoengine import *


class ExceptionError(BaseError):
    """
        This class represents the most typical scenario of error - when an uncaught exception bubbles to the top.
    """

    stacktrace = StringField()

    message = StringField()

