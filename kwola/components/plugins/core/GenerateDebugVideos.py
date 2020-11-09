from ...utils.debug_video import createDebugVideoSubProcess
from ..base.TestingStepPluginBase import TestingStepPluginBase
import atexit
import concurrent.futures
import billiard as multiprocessing
import time


class GenerateDebugVideos(TestingStepPluginBase):
    """
        This plugin creates bug objects for all of the errors discovered during this testing step
    """
    def __init__(self, config):
        self.config = config

    def testingStepStarted(self, testingStep, executionSessions):
        pass

    def beforeActionsRun(self, testingStep, executionSessions, actions):
        pass

    def afterActionsRun(self, testingStep, executionSessions, traces):
        pass

    def testingStepFinished(self, testingStep, executionSessions):
        debugVideoSubprocesses = []

        # Start some parallel processes generating debug videos.
        debugVideoSubprocess1 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(
            self.config.configurationDirectory, str(executionSessions[0].id), "prediction", True, True, None, None, "debug_videos"))
        atexit.register(lambda: debugVideoSubprocess1.terminate())
        debugVideoSubprocesses.append(debugVideoSubprocess1)

        # Leave a gap between the two to reduce collision
        time.sleep(5)

        debugVideoSubprocess2 = multiprocessing.Process(target=createDebugVideoSubProcess, args=(
            self.config.configurationDirectory, str(executionSessions[int(len(executionSessions) / 3)].id), "mix", True, True, None, None, "debug_videos"))
        atexit.register(lambda: debugVideoSubprocess2.terminate())
        debugVideoSubprocesses.append(debugVideoSubprocess2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['video_generation_processes']) as executor:
            futures = []
            for debugVideoSubprocess in debugVideoSubprocesses:
                futures.append(executor.submit(GenerateDebugVideos.runAndJoinSubprocess, debugVideoSubprocess))
            for future in futures:
                future.result()

    def sessionFailed(self, testingStep, executionSession):
        pass

    @staticmethod
    def runAndJoinSubprocess(debugVideoSubprocess):
        debugVideoSubprocess.start()
        debugVideoSubprocess.join()

