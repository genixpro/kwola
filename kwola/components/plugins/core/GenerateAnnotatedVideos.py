from ...utils.debug_video import createDebugVideoSubProcess
from ..base.TestingStepPluginBase import TestingStepPluginBase
import atexit
import concurrent.futures
import multiprocessing


class GenerateAnnotatedVideos(TestingStepPluginBase):
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

        for session in executionSessions:
            debugVideoSubprocess = multiprocessing.Process(target=createDebugVideoSubProcess, args=(self.config.configurationDirectory, str(session.id), "", False, False, None, None, "annotated_videos"))
            atexit.register(lambda: debugVideoSubprocess.terminate() if debugVideoSubprocess is not None else None)
            debugVideoSubprocesses.append(debugVideoSubprocess)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config['video_generation_processes']) as executor:
            futures = []
            for debugVideoSubprocess in debugVideoSubprocesses:
                futures.append(executor.submit(GenerateAnnotatedVideos.runAndJoinSubprocess, debugVideoSubprocess))
            for future in futures:
                future.result()

    def sessionFailed(self, testingStep, executionSession):
        pass

    @staticmethod
    def runAndJoinSubprocess(debugVideoSubprocess):
        debugVideoSubprocess.start()
        debugVideoSubprocess.join()

