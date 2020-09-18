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


from ...datamodels.ActionMapModel import ActionMap
from mongoengine import *


class BaseAction(EmbeddedDocument):
    """
        This model is a base class for all different types of actions that can be performed on a Kwola environment. Actions are
        standard, human driven ways of interacting with the software. They are things you are familiar with,
        such as clicking, typing, tapping, dragging, pinching, and so on.
    """

    meta = {'allow_inheritance': True}


    type = StringField()

    x = FloatField()

    y = FloatField()

    source = StringField()

    # Deprecated
    hadRepeatOverride = FloatField()

    wasRepeatOverride = BooleanField()

    intersectingActionMaps = EmbeddedDocumentListField(ActionMap)

    # Deprecated
    actionMapsAvailable = EmbeddedDocumentListField(ActionMap)

    # Deprecated
    actionMaps = EmbeddedDocumentListField(ActionMap)

    predictedReward = FloatField()

