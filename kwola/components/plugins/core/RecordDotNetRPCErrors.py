from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import os
from kwola.config.logger import getLogger

class RecordDotNetRPCErrors(WebEnvironmentPluginBase):
    def __init__(self):
        self.allErrors = []
        self.allErrorHashes = set()
        self.errorHashes = {}


    def browserSessionStarted(self, webDriver, proxy, executionSession):
        self.errorHashes[executionSession.id] = set()


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        proxy.resetDotNetRPCErrors()


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        for rpcError in proxy.getDotNetRPCErrors():
            rpcError.page = executionTrace.finishURL
            executionTrace.errorsDetected.append(rpcError)
            errorHash = rpcError.computeHash()

            if errorHash not in self.errorHashes[executionSession.id]:
                if errorHash not in self.allErrorHashes and not self.isDuplicate(rpcError):
                    rpcErrorString = f"A dot net RPC error  was detected in client application:\n"
                    rpcErrorString += f"Request: {rpcError.requestData}\n"
                    rpcErrorString += f"Response: {rpcError.responseData}\n"
                    getLogger().info(rpcErrorString)
                    self.allErrorHashes.add(errorHash)
                    self.allErrors.append(rpcError)

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
