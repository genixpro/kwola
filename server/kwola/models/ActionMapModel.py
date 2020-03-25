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

    elementType = StringField()

    def __eq__(self, other):
        if self.elementType != other.elementType:
            return False

        if self.left != other.left:
            return False

        if self.top != other.top:
            return False

        if self.bottom != other.bottom:
            return False

        if self.right != other.right:
            return False

        if self.width != other.width:
            return False

        if self.height != other.height:
            return False

        if self.canClick != other.canClick:
            return False

        if self.canRightClick != other.canRightClick:
            return False

        if self.canType != other.canType:
            return False

        return True

    def __ne__(self, other):
        return not (self == other)
