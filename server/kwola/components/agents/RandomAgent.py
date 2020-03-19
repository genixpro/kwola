from .BaseAgent import BaseAgent
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.actions.RightClickAction import RightClickAction
from kwola.models.actions.TypeAction import TypeAction
from kwola.models.actions.WaitAction import WaitAction
import random

class RandomAgent(BaseAgent):
    """
        This class represents a completely random fuzzer. It will click around totally chaotically and randomly and will
        not do anything remotely intelligent.
    """
    def __init__(self):
        super().__init__()


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

    def nextBestActions(self, stepNumber, images, envActionMaps, additionalFeatures):
        """
            Return the next best action predicted by the agent.
            :param screenshot:
            :return:
        """
        actions = []

        height = images.shape[2]
        width = images.shape[3]

        for sessionN in range(len(images)):

            x = random.randrange(0, width)
            y = random.randrange(0, height)

            actionIndex = random.randrange(0, len(self.actionsSorted))

            action = self.actions[self.actionsSorted[actionIndex]](x=x, y=y)

            action.source = "random"

            actions.append(action)

        return actions

