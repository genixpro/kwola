from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import os
from kwola.config.logger import getLogger

class RecordNetworkErrors(WebEnvironmentPluginBase):
    def __init__(self):
        self.allErrors = []
        self.allErrorHashes = set()
        self.errorHashes = {}


    def browserSessionStarted(self, webDriver, proxy, executionSession):
        self.errorHashes[executionSession.id] = set()


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        proxy.resetNetworkErrors()


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        for networkError in proxy.getNetworkErrors():
            networkError.page = executionTrace.finishURL
            executionTrace.errorsDetected.append(networkError)
            errorHash = networkError.computeHash()

            if errorHash not in self.errorHashes[executionSession.id]:
                if errorHash not in self.allErrorHashes and not self.isDuplicate(networkError):
                    networkErrorMsgString = f"A network error was detected in client application:\n"
                    networkErrorMsgString += f"Path: {networkError.path}\n"
                    networkErrorMsgString += f"Status Code: {networkError.statusCode}\n"
                    networkErrorMsgString += f"Message: {networkError.message}\n"
                    getLogger().info(networkErrorMsgString)
                    self.allErrorHashes.add(errorHash)
                    self.allErrors.append(networkError)

                self.errorHashes[executionSession.id].add(errorHash)
                executionTrace.didNewErrorOccur = True

    def isDuplicate(self, error):
        for existingError in self.allErrors:
            if error.isDuplicateOf(existingError):
                return True
        return False

    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass



    def cleanup(self, webDriver, proxy, executionSession):
        pass
