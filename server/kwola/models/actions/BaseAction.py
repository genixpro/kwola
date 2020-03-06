from mongoengine import *
import datetime


class BaseAction(EmbeddedDocument):
    """
        This model is a base class for all different types of actions that can be performed on a Kwola environment. Actions are
        standard, human driven ways of interacting with the software. They are things you are familiar with,
        such as clicking, typing, tapping, dragging, pinching, and so on.
    """

    meta = {'allow_inheritance': True}


    type = StringField()

