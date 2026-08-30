
import unittest
from ..tasks import TrainAgentLoop
from ..config.config import KwolaCoreConfiguration
import shutil
import os

KROS1_URL = os.environ.get("KWOLA_KROS1_URL", "http://127.0.0.1:3001/")
KROS3_URL = os.environ.get("KWOLA_KROS3_URL", "http://127.0.0.1:3003/")

@unittest.skipUnless(os.environ.get("KWOLA_RUN_KROS_E2E") == "1", "requires the local Kros Compose harness")
class TestTrainingLoop(unittest.TestCase):
    def test_restaurant_click_only(self):
        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url=KROS1_URL,
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
        config['web_session_initialization_timeout'] = 240
        config['web_session_page_load_timeout'] = 180

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)

    def test_restaurant_all_actions(self):
        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url=KROS1_URL,
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
        config['web_session_initialization_timeout'] = 240
        config['web_session_page_load_timeout'] = 180

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)

    def test_kros3_all_actions(self):
        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url=KROS3_URL,
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
        config['web_session_initialization_timeout'] = 240
        config['web_session_page_load_timeout'] = 180

        try:
            TrainAgentLoop.trainAgent(config, exitOnFail=True)
        finally:
            shutil.rmtree(configDir)
