from mongoengine import *


class ActionMap(EmbeddedDocument):
    """
        This class represents a part of the screen you can interact with.
    """

    left = IntField()

    top = IntField()

    bottom = IntField()

    right = IntField()

    width = IntField()

    height = IntField()

    canClick = BooleanField()

    canType = BooleanField()

    canRightClick = BooleanField()
