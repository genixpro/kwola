from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase


class RecordPageURLs(WebEnvironmentPluginBase):
    def __init__(self):
        self.allUrls = {}

    def browserSessionStarted(self, webDriver, proxy, executionSession):
        self.allUrls[executionSession.id] = set()
        self.allUrls[executionSession.id].add(webDriver.current_url)


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        executionTrace.startURL = webDriver.current_url


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        executionTrace.finishURL = webDriver.current_url

        executionTrace.didURLChange = executionTrace.startURL != executionTrace.finishURL
        executionTrace.isURLNew = bool(executionTrace.finishURL not in self.allUrls[executionSession.id])

        self.allUrls[executionSession.id].add(executionTrace.finishURL)

    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass



    def cleanup(self, webDriver, proxy, executionSession):
        pass



