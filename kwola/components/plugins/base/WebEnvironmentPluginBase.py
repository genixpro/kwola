




class WebEnvironmentPluginBase:
    """
        Represents a plugin for the web environments.

        This facilitates hooks that can run before, during, and after
        kwola executes actions on the web-browser
    """

    def browserSessionStarted(self, webDriver, proxy, executionSession):
        pass


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        pass


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        pass


    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass


    def cleanup(self, webDriver, proxy, executionSession):
        pass
