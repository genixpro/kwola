




class TestingStepPluginBase:
    """
        Represents a plugin with hooks on execution of a single testing step.
    """


    def testingStepStarted(self, testingStep, executionSessions):
        pass


    def beforeActionsRun(self, testingStep, executionSessions, actions):
        pass


    def afterActionsRun(self, testingStep, executionSessions, traces):
        pass


    def testingStepFinished(self, testingStep, executionSessions):
        pass


    def sessionFailed(self, testingStep, executionSession):
        pass
