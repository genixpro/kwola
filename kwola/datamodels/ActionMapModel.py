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

    canScroll = BooleanField()

    canScrollUp = BooleanField()

    canScrollDown = BooleanField()

    elementType = StringField()
    
    keywords = StringField()

    inputValue = StringField()

    attributes = DictField(StringField(null=True))

    eventHandlers = ListField(StringField())

    isOnTop = BooleanField()

    def doesOverlapWith(self, other, tolerancePixels=0):
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

        return True

    def isSameAs(self, other, tolerancePixels=0):
        if self.elementType != other.elementType:
            return False

        if not self.doesOverlapWith(other):
            return False

        if self.canClick != other.canClick:
            return False

        if self.canRightClick != other.canRightClick:
            return False

        if self.canType != other.canType:
            return False

        if self.canScroll != other.canScroll:
            return False

        return True

    def canRunAction(self, action):
        from .actions.TypeAction import TypeAction
        from .actions.ClickTapAction import ClickTapAction
        from .actions.RightClickAction import RightClickAction
        from .actions.ScrollingAction import ScrollingAction
        from .actions.ClearFieldAction import ClearFieldAction

        if isinstance(action, TypeAction):
            if self.canType:
                return True

        if isinstance(action, ClearFieldAction):
            if self.canType:
                return True

        if isinstance(action, ClickTapAction):
            if self.canClick:
                return True

        if isinstance(action, RightClickAction):
            if self.canRightClick:
                return True

        if isinstance(action, ScrollingAction):
            if self.canScroll:
                return True

        return False

    def __eq__(self, other):
        return self.doesOverlapWith(other)

    def __ne__(self, other):
        return not (self == other)
