from kwola.config.logger import getLogger
from ..base.TestingStepPluginBase import TestingStepPluginBase
import numpy
import os




class LogSessionActionExecutionTimes(TestingStepPluginBase):
    """
        This plugin creates bug objects for all of the errors discovered during this testing step
    """
    def __init__(self, config):
        self.config = config

        self.listOfTimesForScreenshot = []
        self.listOfTimesForActionMapRetrieval = []
        self.listOfTimesForActionDecision = []
        self.listOfTimesForActionExecution = []
        self.listOfTimesForMiscellaneous = []
        self.listOfTotalLoopTimes = []

    def testingStepStarted(self, testingStep, executionSessions):
        pass

    def beforeActionsRun(self, testingStep, executionSessions, actions):
        pass

    def afterActionsRun(self, testingStep, executionSessions, traces):
        trace = [trace for trace in traces if trace is not None][0]

        self.listOfTimesForScreenshot.append(trace.timeForScreenshot)
        self.listOfTimesForActionMapRetrieval.append(trace.timeForActionMapRetrieval)
        self.listOfTimesForActionDecision.append(trace.timeForActionDecision)
        self.listOfTimesForActionExecution.append(trace.timeForActionExecution)
        self.listOfTimesForMiscellaneous.append(trace.timeForMiscellaneous)

        totalTime = trace.timeForScreenshot + \
            trace.timeForActionMapRetrieval + \
            trace.timeForActionDecision + \
            trace.timeForActionExecution + \
            trace.timeForMiscellaneous

        self.listOfTotalLoopTimes.append(totalTime)

        if trace.traceNumber % self.config['testing_print_every'] == (self.config['testing_print_every'] - 1) or trace.traceNumber == 0:
            msg = f"Finished {trace.traceNumber + 1} testing actions."
            if len(self.listOfTimesForScreenshot):
                msg += f"\n     Avg Screenshot time: {numpy.average(self.listOfTimesForScreenshot[-self.config['testing_print_every']:]):.4f}"
                msg += f"\n     Avg Action Map Retrieval Time: {numpy.average(self.listOfTimesForActionMapRetrieval[-self.config['testing_print_every']:]):.4f}"
                msg += f"\n     Avg Action Decision Time: {numpy.average(self.listOfTimesForActionDecision[-self.config['testing_print_every']:]):.4f}"
                msg += f"\n     Avg Action Execution Time: {numpy.average(self.listOfTimesForActionExecution[-self.config['testing_print_every']:]):.4f}"
                msg += f"\n     Avg Miscellaneous Time: {numpy.average(self.listOfTimesForMiscellaneous[-self.config['testing_print_every']:]):.4f}"
                msg += f"\n     Avg Total Loop Time: {numpy.average(self.listOfTotalLoopTimes[-self.config['testing_print_every']:]):.4f}"
            getLogger().info(msg)

    def testingStepFinished(self, testingStep, executionSessions):
        pass

    def sessionFailed(self, testingStep, executionSession):
        pass
