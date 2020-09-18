from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
import selenium.common.exceptions


class RecordCursorAtAction(WebEnvironmentPluginBase):
    def browserSessionStarted(self, webDriver, proxy, executionSession):
        pass


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        try:
            element = webDriver.execute_script("""
            return document.elementFromPoint(arguments[0], arguments[1]);
            """, actionToExecute.x, actionToExecute.y)

            if element is not None:
                executionTrace.cursor = element.value_of_css_property("cursor")
            else:
                executionTrace.cursor = None

        except selenium.common.exceptions.StaleElementReferenceException as e:
            executionTrace.cursor = None


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        pass


    def browserSessionFinished(self, webDriver, proxy, executionSession):
        pass



    def cleanup(self, webDriver, proxy, executionSession):
        pass



