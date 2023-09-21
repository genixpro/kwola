#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from ..tasks import TrainAgentLoop
from .main import getConfigurationDirFromCommandLineArgs
from ..diagnostics.test_installation import testInstallation
from ..config.logger import getLogger, setupLocalLogging
from ..config.config import KwolaCoreConfiguration
import logging

def main():
    """
        This is the entry point for the Kwola secondary command, kwola_train_agent.

        It is basically identical in behaviour as the main function.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print("Refusing to start training loop. There appears to be a problem with your Kwola installation or environment.")
        exit(1)

    configDir = getConfigurationDirFromCommandLineArgs()
    config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)
    TrainAgentLoop.trainAgent(config)
