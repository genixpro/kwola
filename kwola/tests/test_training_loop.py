
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
                                                                        web_session_autologin=True,
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
                                                                        actions_custom_typing_action_strings=[],
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
                                                                        web_session_autologin=True,
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
                                                                        actions_custom_typing_action_strings=[
                                                                            'action_a',
                                                                            'b_action'
                                                                        ]
                                                                        )

        config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)

    def test_kros3_all_actions(self):
        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url="http://kros3.kwola.io/",
                                                                        email=None,
                                                                        password=None,
                                                                        web_session_autologin=False,
                                                                        name=None,
                                                                        paragraph=None,
                                                                        enableRandomEmailCommand=True,
                                                                        enableScrolling=True,
                                                                        enableTypeEmail=False,
                                                                        enableTypePassword=False,
                                                                        enableRandomNumberCommand=False,
                                                                        enableRandomBracketCommand=False,
                                                                        enableRandomMathCommand=False,
                                                                        enableRandomOtherSymbolCommand=False,
                                                                        enableDoubleClickCommand=False,
                                                                        enableRightClickCommand=False,
                                                                        actions_custom_typing_action_strings=["test1", "test2", "test3", "test4"],
                                                                        web_session_no_network_activity_wait_time=0.0,
                                                                        web_session_perform_action_wait_time=0.1,
                                                                        web_session_initial_fetch_sleep_time=1
                                                                        )

        config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)
