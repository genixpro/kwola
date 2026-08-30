
import unittest
from ..components.environments.WebEnvironment import WebEnvironment
from ..datamodels.ExecutionSessionModel import ExecutionSession
from ..datamodels.ExecutionTraceModel import ExecutionTrace
from ..config.config import KwolaCoreConfiguration
from ..components.plugins.core.RecordPageHTML import RecordPageHTML
from datetime import datetime
import shutil
import traceback
from ..config.logger import getLogger, setupLocalLogging
import cProfile
import pstats
import os

@unittest.skipUnless(os.environ.get("KWOLA_RUN_KROS_E2E") == "1", "requires the local Kros Compose harness")
class TestHTMLSaver(unittest.TestCase):
    def test_html_saving(self):
        browser = os.environ.get("KWOLA_TEST_BROWSER", "chrome")
        configDir = KwolaCoreConfiguration.createNewLocalKwolaConfigDir("testing",
                                                                        url=os.environ.get("KWOLA_KROS1_URL", "http://127.0.0.1:3001/"),
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

        try:
            config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)
            # Kros 1's frozen Angular/Bower bundle is instrumented on its
            # first load.  Allow that cold rewrite to finish rather than
            # letting the generic interactive-session timeout start retries.
            config['web_session_initialization_timeout'] = 240
            config['web_session_page_load_timeout'] = 180
            config['enable_record_page_html'] = True

            session = ExecutionSession(
                id="html_save_test",
                owner="testing",
                status="running",
                testingStepId=None,
                testingRunId=None,
                applicationId=None,
                startTime=datetime.now(),
                endTime=None,
                tabNumber=0,
                executionTraces=[],
                browser=browser,
                windowSize="desktop"
            )

            executionTrace = ExecutionTrace(id=str(session.id) + "-trace-0")

            environment = WebEnvironment(config=config, sessionLimit=1, executionSessions=[session], plugins=[], browser=browser, windowSize="desktop")
            environmentSession = environment.sessions[0]

            htmlPlugin = [plugin for plugin in environment.plugins if isinstance(plugin, RecordPageHTML)][0]

            profile = cProfile.Profile()
            profile.enable()
            start = datetime.now()
            htmlPlugin.saveHTML(environmentSession.driver, environmentSession.proxy, executionTrace)
            end = datetime.now()
            #
            profile.disable()

            stats = pstats.Stats(profile).sort_stats("cumtime")
            stats.print_stats()
            stats.print_callers()


            print(f"{(end - start).total_seconds()} total seconds to save html")

            if environment.sessions[0].browserDeathReason:
                print(environment.sessions[0].browserDeathReason)

            environment.shutdown()
        except Exception:
            getLogger().error(f"{traceback.format_exc()}")
            raise
        finally:
            shutil.rmtree(configDir)
