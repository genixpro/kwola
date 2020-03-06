from .BaseAction import BaseAction
from mongoengine import *


class ClickTapAction(BaseAction):
    """
        This class represents a click / tap action. Clicks are used when testing web-applications and taps are
        used when testing mobile.
    """

    x = FloatField()
    y = FloatField()


