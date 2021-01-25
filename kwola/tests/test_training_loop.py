
import unittest
from ..tasks import TrainAgentLoop
from ..config.config import KwolaCoreConfiguration
import shutil

class TestTrainingLoop(unittest.TestCase):
    def test_restaurant_click_only(self):
        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url="http://kros1.kwola.io/",
                                                                        email="test1@test.com",
                                                                        password="test1",
                                                                        autologin=True,
                                                                        name="",
                                                                        paragraph="",
                                                                        enableTypeEmail=True,
                                                                        enableTypePassword=True,
                                                                        enableRandomNumberCommand=False,
                                                                        enableRandomBracketCommand=False,
                                                                        enableRandomMathCommand=False,
                                                                        enableRandomOtherSymbolCommand=False,
                                                                        enableDoubleClickCommand=False,
                                                                        enableRightClickCommand=False,
                                                                        custom_typing_action_strings=[],
                                                                        enableScrolling=True
                                                                        )

        config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)

    def test_restaurant_all_actions(self):
        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url="http://kros1.kwola.io/",
                                                                        email="test1@test.com",
                                                                        password="test1",
                                                                        autologin=True,
                                                                        name="Kwola",
                                                                        paragraph="Kwola is the shit. You should try it out now.",
                                                                        enableTypeEmail=True,
                                                                        enableTypePassword=True,
                                                                        enableRandomNumberCommand=True,
                                                                        enableRandomBracketCommand=True,
                                                                        enableRandomMathCommand=True,
                                                                        enableRandomOtherSymbolCommand=True,
                                                                        enableDoubleClickCommand=True,
                                                                        enableRightClickCommand=True,
                                                                        custom_typing_action_strings=[
                                                                            'action_a',
                                                                            'b_action'
                                                                        ]
                                                                        )

        config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)
