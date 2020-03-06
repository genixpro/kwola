from .BaseAction import BaseAction
from mongoengine import *


class WaitAction(BaseAction):
    """
        This class represents when you do nothing for some period of time before performing the next action.
    """

    time = FloatField()



