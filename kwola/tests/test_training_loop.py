
import unittest
from ..tasks import TrainAgentLoop
from ..config.config import Configuration
import shutil

class TestTrainingLoop(unittest.TestCase):
    def test_restaurant_click_only(self):
        configDir = Configuration.createNewLocalKwolaConfigDir("testing",
                                                               url="http://demo.kwolatesting.com/",
                                                               email="",
                                                               password="",
                                                               name="",
                                                               paragraph="",
                                                               enableRandomNumberCommand=False,
                                                               enableRandomBracketCommand=False,
                                                               enableRandomMathCommand=False,
                                                               enableRandomOtherSymbolCommand=False,
                                                               enableDoubleClickCommand=False,
                                                               enableRightClickCommand=False
                                                               )
        try:
            TrainAgentLoop.trainAgent(configDir, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)

    def test_restaurant_all_actions(self):
        configDir = Configuration.createNewLocalKwolaConfigDir("testing",
                                                               url="http://demo.kwolatesting.com/",
                                                               email="test1@test.com",
                                                               password="test1",
                                                               name="Kwola",
                                                               paragraph="Kwola is the shit. You should try it out now.",
                                                               enableRandomNumberCommand=True,
                                                               enableRandomBracketCommand=True,
                                                               enableRandomMathCommand=True,
                                                               enableRandomOtherSymbolCommand=True,
                                                               enableDoubleClickCommand=True,
                                                               enableRightClickCommand=True
                                                               )
        try:
            TrainAgentLoop.trainAgent(configDir, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)
