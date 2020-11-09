from kwola.config.logger import getLogger
from ..base.TestingStepPluginBase import TestingStepPluginBase
import numpy
import os




class LogSessionRewards(TestingStepPluginBase):
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
        totalRewards = []
        for session in executionSessions:
            getLogger().info(
                f"Session {session.tabNumber} finished with total reward: {session.totalReward:.3f}")
            totalRewards.append(session.totalReward)

        if len(totalRewards) > 0:
            getLogger().info(f"Mean total reward of all sessions: {numpy.mean(totalRewards):.3f}")

    def sessionFailed(self, testingStep, executionSession):
        pass
