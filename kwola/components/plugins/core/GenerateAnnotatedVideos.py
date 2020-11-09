from ...utils.debug_video import createDebugVideoSubProcess
from ..base.TestingStepPluginBase import TestingStepPluginBase
import atexit
import concurrent.futures
import billiard as multiprocessing


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
        pool = multiprocessing.Pool(self.config['video_generation_processes'], maxtasksperchild=1)

        futures = []
        for session in executionSessions:
            future = pool.apply_async(func=createDebugVideoSubProcess, args=(self.config.configurationDirectory, str(session.id), "", False, False, None, None, "annotated_videos"))
            futures.append(future)

        for future in futures:
            future.get()

        pool.close()
        pool.join()

    def sessionFailed(self, testingStep, executionSession):
        pass
