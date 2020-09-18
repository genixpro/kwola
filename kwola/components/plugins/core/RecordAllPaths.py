from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import selenium.common.exceptions


class RecordAllPaths(WebEnvironmentPluginBase):
    def __init__(self):
        self.lastProxyPaths = {}

    def browserSessionStarted(self, webDriver, proxy, executionSession):
        self.lastProxyPaths[executionSession.id] = set()


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        proxy.resetPathTrace()


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        urlPathTrace = proxy.getPathTrace()

        cumulativeProxyPaths = urlPathTrace['seen']
        newProxyPaths = cumulativeProxyPaths.difference(self.lastProxyPaths[executionSession.id])

        executionTrace.networkTrafficTrace = list(urlPathTrace['recent'])
        executionTrace.hadNetworkTraffic = len(urlPathTrace['recent']) > 0
        executionTrace.hadNewNetworkTraffic = len(newProxyPaths) > 0

        self.lastProxyPaths[executionSession.id] = set(urlPathTrace['seen'])

    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass



    def cleanup(self, webDriver, proxy, executionSession):
        pass



