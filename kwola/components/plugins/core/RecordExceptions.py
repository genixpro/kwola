from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import selenium.common.exceptions
import os
from kwola.config.logger import getLogger
from kwola.datamodels.errors.ExceptionError import ExceptionError
from .common import kwolaJSRewriteErrorDetectionStrings

class RecordExceptions(WebEnvironmentPluginBase):

    def __init__(self):
        self.allErrorHashes = set()
        self.allErrors = []
        self.errorHashes = {}


    def browserSessionStarted(self, webDriver, proxy, executionSession):
        self.errorHashes[executionSession.id] = set()

    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        # Inject bug detection script
        webDriver.execute_script("""
            if (!window.kwolaExceptions)
            {
                window.kwolaExceptions = [];
                var kwolaCurrentOnError = window.onerror;
                window.onerror=function(msg, source, lineno, colno, error) {
                    let stack = null;
                    if (error)
                    {
                        stack = error.stack;
                    }
    
                    window.kwolaExceptions.push([msg, source, lineno, colno, stack]);
                    if (kwolaCurrentOnError)
                    {
                        kwolaCurrentOnError(msg, source, lineno, colno, error);
                    }
                }
            }
        """)



    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        exceptions = self.extractExceptions(webDriver)

        for exception in exceptions:
            msg, source, lineno, colno, stack = tuple(exception)

            msg = str(msg)
            source = str(source)
            stack = str(stack)

            combinedMessage = msg + source + stack

            kwolaJSRewriteErrorFound = False
            for detectionString in kwolaJSRewriteErrorDetectionStrings:
                if detectionString in combinedMessage:
                    kwolaJSRewriteErrorFound = True

            if kwolaJSRewriteErrorFound:
                logMsgString = f"Error. There was a bug generated by the underlying javascript application, " \
                               f"but it appears to be a bug in Kwola's JS rewriting. Please notify the Kwola " \
                               f"developers that this url: {webDriver.current_url} gave you a js-code-rewriting " \
                               f"issue. \n"

                logMsgString += f"{msg} at line {lineno} column {colno} in {source}\n"

                logMsgString += f"{str(stack)}\n"

                getLogger().error(logMsgString)
            else:
                error = ExceptionError(type="exception", page=executionTrace.finishURL, stacktrace=stack, message=msg,
                                       source=source, lineNumber=lineno, columnNumber=colno)
                executionTrace.errorsDetected.append(error)
                errorHash = error.computeHash()

                executionTrace.didErrorOccur = True

                if errorHash not in self.errorHashes[executionSession.id]:
                    if errorHash not in self.allErrorHashes and not self.isDuplicate(error):
                        logMsgString = f"An unhandled exception was detected in client application:\n"
                        logMsgString += f"{msg} at line {lineno} column {colno} in {source}\n"
                        logMsgString += f"{str(stack)}"
                        getLogger().info(logMsgString)
                        self.allErrorHashes.add(errorHash)
                        self.allErrors.append(error)

                    self.errorHashes[executionSession.id].add(errorHash)
                    executionTrace.didNewErrorOccur = True


    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass

    def isDuplicate(self, error):
        for existingError in self.allErrors:
            if error.isDuplicateOf(existingError):
                return True
        return False

    def extractExceptions(self, webDriver):
        # The JavaScript that we want to inject. This will extract out the exceptions
        # that the Kwola error handler was able to pick up
        injected_javascript = (
            'const exceptions = window.kwolaExceptions; window.kwolaExceptions = []; return exceptions;'
        )

        result = webDriver.execute_script(injected_javascript)

        if result is None:
            return []

        return result



    def cleanup(self, webDriver, proxy, executionSession):
        pass