
import unittest
from ..tasks import TrainAgentLoop
from ..config.config import Configuration



class TestTrainingLoop(unittest.TestCase):

    def test_restaurant(self):
        configDir = Configuration.createNewLocalKwolaConfigDir("testing",
                                                               url="http://demo.kwolatesting.com/",
                                                               email="test1@test.com",
                                                               password="test1",
                                                               name="",
                                                               paragraph="",
                                                               enableRandomNumberCommand=False,
                                                               enableRandomBracketCommand=False,
                                                               enableRandomMathCommand=False,
                                                               enableRandomOtherSymbolCommand=False,
                                                               enableDoubleClickCommand=False,
                                                               enableRightClickCommand=False
                                                               )
        TrainAgentLoop.trainAgent(configDir)

