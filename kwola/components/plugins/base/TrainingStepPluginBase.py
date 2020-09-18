




class TrainingStepPluginBase:
    """
        Represents a plugin with hooks that execute while the model is training.
    """


    def trainingStepStarted(self, trainingStep):
        pass


    def iterationCompleted(self, trainingStep):
        pass


    def trainingStepFinished(self, trainingStep):
        pass


