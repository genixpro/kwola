from kwola.components.plugins.base.WebEnvironmentPluginBase import WebEnvironmentPluginBase
from kwola.config.logger import getLogger



class RecordFitness(WebEnvironmentPluginBase):
    def __init__(self):
        self.fitnessValues = {}

    def browserSessionStarted(self, webDriver, proxy, executionSession):
        pass


    def beforeActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionToExecute):
        executionTrace.startApplicationProvidedCumulativeFitness = self.extractFitness(webDriver)

        if executionTrace.startApplicationProvidedCumulativeFitness is not None:
            if executionSession.bestApplicationProvidedCumulativeFitness is None:
                executionSession.bestApplicationProvidedCumulativeFitness = executionTrace.startApplicationProvidedCumulativeFitness
            else:
                executionSession.bestApplicationProvidedCumulativeFitness = max(executionSession.bestApplicationProvidedCumulativeFitness, executionTrace.startApplicationProvidedCumulativeFitness)


    def afterActionRuns(self, webDriver, proxy, executionSession, executionTrace, actionExecuted):
        executionTrace.endApplicationProvidedCumulativeFitness = self.extractFitness(webDriver)

        if executionTrace.endApplicationProvidedCumulativeFitness is not None:
            if executionSession.bestApplicationProvidedCumulativeFitness is None:
                executionSession.bestApplicationProvidedCumulativeFitness = executionTrace.endApplicationProvidedCumulativeFitness
            else:
                executionSession.bestApplicationProvidedCumulativeFitness = max(executionSession.bestApplicationProvidedCumulativeFitness, executionTrace.endApplicationProvidedCumulativeFitness)


    def browserSessionFinished(self, webDriver, proxy, executionSession):
        if executionSession.bestApplicationProvidedCumulativeFitness is not None:
            getLogger().info(f"Session {executionSession.id} finished with fitness: {executionSession.bestApplicationProvidedCumulativeFitness:.0f}")

    def extractFitness(self, webDriver):
        injected_javascript = (
            'return window.kwolaCumulativeFitness;'
        )

        result = webDriver.execute_script(injected_javascript)

        return result



    def cleanup(self, webDriver, proxy, executionSession):
        pass
