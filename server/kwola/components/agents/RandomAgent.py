from .BaseAgent import BaseAgent
from kwola.models.actions.ClickTapAction import ClickTapAction
import random

class RandomAgent(BaseAgent):
    """
        This class represents a completely random fuzzer. It will click around totally chaotically and randomly and will
        not do anything remotely intelligent.
    """
    def __init__(self):
        pass



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


    def initialize(self, environment):
        """
        Initialize the agent for operating in the given environment.

        :param environment:
        :return:
        """

    def nextBestAction(self, screenshot):
        """
            Return the next best action predicted by the agent.
            :param screenshot:
            :return:
        """
        x = random.randint(500)
        y = random.randint(500)

        action = ClickTapAction(x=x, y=y)

        return action


    def learnFromTestingSequence(self, testingSequence):
        """
            Runs the backward pass / gradient update so the algorithm can learn from all the memories in the given testing sequence

            :param testingSequence:
            :return:
        """

