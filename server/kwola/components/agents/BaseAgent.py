



class BaseAgent:
    """
        This is a base class for different implementations of the machine learning algorithms that are used to
        control the Environments and find bugs.


    """


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

    def learnFromTestingSequence(self, testingSequence):
        """
            Runs the backward pass / gradient update so the algorithm can learn from all the memories in the given testing sequence

            :param testingSequence:
            :return:
        """

