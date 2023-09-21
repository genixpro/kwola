#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .BaseAction import BaseAction
from mongoengine import *


class ScrollingAction(BaseAction):
    """
        This class represents a scroll down.
    """

    direction = StringField()

