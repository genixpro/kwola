#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from ..config.config import KwolaCoreConfiguration
from ..datamodels.CustomIDField import CustomIDField
from ..datamodels.TestingStepModel import TestingStep
from ..tasks import RunTestingStep
from .main import getConfigurationDirFromCommandLineArgs
from ..diagnostics.test_installation import testInstallation
from ..config.logger import getLogger, setupLocalLogging
import logging

def main():
    """
        This is the entry point for the Kwola secondary command, kwola_run_test_step.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print("Refusing to start testing step. There appears to be a problem with your Kwola installation or environment.")
        exit(1)

    configDir = getConfigurationDirFromCommandLineArgs()
    config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

    testingStep = TestingStep(id=CustomIDField.generateNewUUID(TestingStep, config), testStepIndexWithinRun=0)
    testingStep.saveToDisk(config)

    RunTestingStep.runTestingStep(configDir, str(testingStep.id))
