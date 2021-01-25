from ...utils.debug_video import createDebugVideoSubProcess
from ...utils.retry import autoretry
from ..base.TestingStepPluginBase import TestingStepPluginBase
import atexit
import concurrent.futures
import billiard as multiprocessing
import billiard.exceptions
from kwola.config.logger import getLogger


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

    @autoretry(exponentialBackOffBase=3)
    def testingStepFinished(self, testingStep, executionSessions):
        pool = multiprocessing.Pool(self.config['video_generation_processes'], maxtasksperchild=1)

        futures = []
        for session in executionSessions:
            future = pool.apply_async(func=createDebugVideoSubProcess, args=(self.config.serialize(), str(session.id), "", False, False, None, None, "annotated_videos"))
            futures.append((session, future))

        for session, future in futures:
            localFuture = future
            # for retry in range(5):
            # try:
            value = localFuture.get(timeout=self.config['debug_video_generation_timeout'])
            if value:
                getLogger().error(value)
            # break
            # except billiard.exceptions.WorkerLostError:
            #     if retry == 4:
            #         raise
            #     localFuture = pool.apply_async(func=createDebugVideoSubProcess,
            #                               args=(self.config.serialize(), str(session.id), "", False, False, None, None, "annotated_videos"))
            # except BrokenPipeError:
            #     if retry == 4:
            #         raise
            #     localFuture = pool.apply_async(func=createDebugVideoSubProcess,
            #                               args=(self.config.serialize(), str(session.id), "", False, False, None, None, "annotated_videos"))





        pool.close()
        pool.join()

    def sessionFailed(self, testingStep, executionSession):
        pass
