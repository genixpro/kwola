#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
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

