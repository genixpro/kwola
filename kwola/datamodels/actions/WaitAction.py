#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .BaseAction import BaseAction
from mongoengine import *


class WaitAction(BaseAction):
    """
        This class represents when you do nothing for some period of time before performing the next action.
    """

    time = FloatField()



