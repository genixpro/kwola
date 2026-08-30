from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
from kwola.components.environments.PlaywrightBrowserSession import PlaywrightError


class RecordCursorAtAction(WebEnvironmentPluginBase):
    def browserSessionStarted(self, webDriver, proxy, executionSession):
        pass


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        try:
            executionTrace.cursor = webDriver.css_property_at(actionToExecute.x, actionToExecute.y, "cursor")

        except PlaywrightError:
            executionTrace.cursor = None


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        pass


    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass



    def cleanup(self, webDriver, proxy, executionSession):
        pass


