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
    
    keywords = StringField()

    def doesOverlapWith(self, other, tolerancePixels=0):
        if self.elementType != other.elementType:
            return False

        if abs(self.left - other.left) > tolerancePixels:
            return False

        if abs(self.top - other.top) > tolerancePixels:
            return False

        if abs(self.bottom - other.bottom) > tolerancePixels:
            return False

        if abs(self.right - other.right) > tolerancePixels:
            return False

        if abs(self.width - other.width) > tolerancePixels:
            return False

        if abs(self.height - other.height) > tolerancePixels:
            return False

        if self.canClick != other.canClick:
            return False

        if self.canRightClick != other.canRightClick:
            return False

        if self.canType != other.canType:
            return False

        return True

    def __eq__(self, other):
        return self.doesOverlapWith(other)

    def __ne__(self, other):
        return not (self == other)
