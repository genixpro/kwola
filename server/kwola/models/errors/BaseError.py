from mongoengine import *
import datetime


class BaseError(EmbeddedDocument):
    """
        This model is a base class for all different kinds of errors that can be detected by the Kwola engine.
    """

    meta = {'allow_inheritance': True}

    type = StringField()

