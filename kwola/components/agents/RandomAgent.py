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


from .BaseAgent import BaseAgent
from ...models.actions.ClickTapAction import ClickTapAction
from ...models.actions.RightClickAction import RightClickAction
from ...models.actions.TypeAction import TypeAction
from ...models.actions.WaitAction import WaitAction
import random
import numpy

class RandomAgent(BaseAgent):
    """
        This class represents a completely random fuzzer. It will click around totally chaotically and randomly and will
        not do anything remotely intelligent.
    """
    def __init__(self, config):
        super().__init__(config)


    def load(self):
        """
            Loads the agent from db / disk

            :return:
        """


    def save(self):
        """
            Saves the agent to the db / disk.

            :return:
        """


    def initialize(self, branchFeatureSize):
        """
        Initialize the agent for operating in the given environment.

        :param branchFeatureSize:
        :return:
        """

        self.branchFeatureSize = branchFeatureSize

    def nextBestActions(self, stepNumber, rawImages, envActionMaps, additionalFeatures, recentActions):
        """
            Return the next best action predicted by the agent.
            :param screenshot:
            :return:
        """
        actions = []

        images = numpy.array(images)

        height = images.shape[1]
        width = images.shape[2]

        for sessionN in range(len(images)):

            x = random.randrange(0, width)
            y = random.randrange(0, height)

            actionIndex = random.randrange(0, len(self.actionsSorted))

            action = self.actions[self.actionsSorted[actionIndex]](x=x, y=y)

            action.source = "random"

            actions.append(action)

        return actions

