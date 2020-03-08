
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.actions.RightClickAction import RightClickAction
from kwola.models.actions.TypeAction import TypeAction
from kwola.models.actions.WaitAction import WaitAction



class BaseAgent:
    """
        This is a base class for different implementations of the machine learning algorithms that are used to
        control the Environments and find bugs.


    """

    def __init__(self):


        self.actions = {
            "click": lambda x,y: ClickTapAction(type="click", x=x, y=y, times=1),
            # "double_click": lambda x,y: ClickTapAction(type="double_click", x=x, y=y, times=2),
            # "right_click": lambda x,y: RightClickAction(type="right_click", x=x, y=y),
            # "wait": lambda x,y: WaitAction(type="wait", x=x, y=y, time=3),
            "typeEmail": lambda x,y: TypeAction(type="typeEmail", x=x, y=y, label="email", text="test1@test.com"),
            "typePassword": lambda x,y: TypeAction(type="typePassword", x=x, y=y, label="password", text="test1"),
            # "typeName": lambda x,y: TypeAction(type="typeName", x=x, y=y, label="name", text="Brad"),
            # "typeParagraph": lambda x,y: TypeAction(type="typeParagraph", x=x, y=y, label="paragraph", text="Dolore prodesset incorrupte duo te, natum pericula te sea. Vis no vero ludus noster, eu eum eros nusquam inciderint, his in elit possit torquatos. Ne est eros expetenda, ne quis nostrum vis. His in scripserit signiferumque, ut minim harum graece nam. At errem noluisse partiendo per, nec ex accusata dissentiunt. Simul populo appareat cu quo, dicam prompta virtute eu nec, nobis pertinax in nam.")
        }

        self.actionsSorted = sorted(self.actions.keys())



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

