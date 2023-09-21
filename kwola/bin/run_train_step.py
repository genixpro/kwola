#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from ..config.config import KwolaCoreConfiguration
from ..datamodels.CustomIDField import CustomIDField
from ..datamodels.TrainingStepModel import TrainingStep
from ..tasks import RunTrainingStep
from .main import getConfigurationDirFromCommandLineArgs
from ..diagnostics.test_installation import testInstallation
from ..config.logger import getLogger, setupLocalLogging
import logging

def main():
    """
        This is the entry point for the Kwola secondary command, kwola_run_train_step.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print("Refusing to run training step. There appears to be a problem with your Kwola installation or environment.")
        exit(1)

    configDir = getConfigurationDirFromCommandLineArgs()
    config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

    trainingStep = TrainingStep(id=CustomIDField.generateNewUUID(TrainingStep, config))
    trainingStep.saveToDisk(config)

    RunTrainingStep.runTrainingStep(configDir, str(trainingStep.id), 0)
