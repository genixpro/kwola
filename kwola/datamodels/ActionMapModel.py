#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


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
