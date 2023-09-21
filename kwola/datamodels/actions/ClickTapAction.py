#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .BaseAction import BaseAction
from mongoengine import *


class ClickTapAction(BaseAction):
    """
        This class represents a click / tap action. Clicks are used when testing web-applications and taps are
        used when testing mobile.
    """

    times = IntField() # Number of times to click / tap in one spot. Can be 1, 2 or 3`

