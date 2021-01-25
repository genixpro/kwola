from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import selenium.common.exceptions
from kwola.config.logger import getLogger
import numpy
import os
from selenium.webdriver import Firefox, Chrome, Edge



class RecordBranchTrace(WebEnvironmentPluginBase):
    def __init__(self):
        self.cumulativeBranchTrace = {}

    def browserSessionStarted(self, webDriver, proxy, executionSession):
        self.cumulativeBranchTrace[executionSession.id] = self.extractBranchTrace(webDriver)


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        pass


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        branchTrace = self.extractBranchTrace(webDriver)

        newBranches = False
        filteredBranchTrace = {}

        for fileName in branchTrace.keys():
            traceVector = branchTrace[fileName]
            didExecuteFile = bool(numpy.sum(traceVector) > 0)

            if didExecuteFile:
                filteredBranchTrace[fileName] = traceVector

            if fileName in self.cumulativeBranchTrace[executionSession.id]:
                cumulativeTraceVector = self.cumulativeBranchTrace[executionSession.id][fileName]

                if traceVector.shape[0] == cumulativeTraceVector.shape[0]:
                    newBranchCount = numpy.sum(traceVector[cumulativeTraceVector == 0])
                    if newBranchCount > 0:
                        newBranches = True
                else:
                    if didExecuteFile:
                        newBranches = True
            else:
                if didExecuteFile:
                    newBranches = True

        executionTrace.branchTrace = {k: v.tolist() for k, v in filteredBranchTrace.items()}

        executionTrace.didCodeExecute = bool(len(filteredBranchTrace) > 0)
        executionTrace.didNewBranchesExecute = bool(newBranches)

        total = 0
        executedAtleastOnce = 0
        for fileName in self.cumulativeBranchTrace[executionSession.id]:
            total += self.cumulativeBranchTrace[executionSession.id][fileName].shape[0]
            executedAtleastOnce += numpy.count_nonzero(self.cumulativeBranchTrace[executionSession.id][fileName])

        # Just an extra check here to cover our ass in case of division by zero
        if total == 0:
            total += 1

        executionTrace.cumulativeBranchCoverage = float(executedAtleastOnce) / float(total)

        for fileName in filteredBranchTrace.keys():
            if fileName in self.cumulativeBranchTrace[executionSession.id]:
                if branchTrace[fileName].shape[0] == self.cumulativeBranchTrace[executionSession.id][fileName].shape[0]:
                    self.cumulativeBranchTrace[executionSession.id][fileName] += branchTrace[fileName]
                else:
                    getLogger().warning(
                        f"Warning! The file with fileName {fileName} has changed the size of its trace vector. This "
                        f"is very unusual and could indicate some strange situation with dynamically loaded javascript")
            else:
                self.cumulativeBranchTrace[executionSession.id][fileName] = branchTrace[fileName]

    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass

    def extractBranchTrace(self, webDriver):
        # The JavaScript that we want to inject. This will extract out the Kwola debug information.
        if isinstance(webDriver, Firefox):
            injected_javascript = (
                'const newCounters = {};'
                'if (window.kwolaCounters)'
                '{'
                '   for (const [key, value] of Object.entries(window.kwolaCounters))'
                '   {'
                '       newCounters[key] = Array.from(value);'
                '   }'
                '   return newCounters;'
                '}'
                'else'
                '{'
                '    return null;'
                '}'
            )
        elif isinstance(webDriver, Chrome) or isinstance(webDriver, Edge):
            injected_javascript = (
                'return window.kwolaCounters;'
            )
        else:
            raise RuntimeError("Unrecognized web driver class.")

        result = webDriver.execute_script(injected_javascript)

        # The JavaScript that we want to inject. This will extract out the Kwola debug information.
        injected_javascript = (
            'if (!window.kwolaCounters)'
            '{'
            '   window.kwolaCounters = {};'
            '}'
            'Object.keys(window.kwolaCounters).forEach((fileName) => {'
            '   window.kwolaCounters[fileName].fill(0);'
            '});'
        )

        try:
            webDriver.execute_script(injected_javascript)
        except selenium.common.exceptions.TimeoutException:
            getLogger().warning(f"Warning, timeout while running the script to reset the kwola line counters.")

        if result is not None:
            # Cast everything to a numpy array so we don't have to do it later
            for fileName, vector in result.items():
                result[fileName] = numpy.array(vector)
        else:
            getLogger().warning(f"Warning, did not find the kwola line counter object in the browser. This usually "
                  "indicates that there was an error either in translating the javascript, an error "
                  "in loading the page, or that the page has absolutely no javascript. "
                  f"On page: {webDriver.current_url}")
            result = {}

        return result



    def cleanup(self, webDriver, proxy, executionSession):
        pass
