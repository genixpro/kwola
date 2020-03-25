from mongoengine import *
import datetime
from kwola.models.ActionMapModel import ActionMap


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

    actionMapsAvailable = EmbeddedDocumentListField(ActionMap)

    # Deprecated
    actionMaps = EmbeddedDocumentListField(ActionMap)

    predictedReward = FloatField()

