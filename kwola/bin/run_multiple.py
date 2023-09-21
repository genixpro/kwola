#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from ..config.config import KwolaCoreConfiguration
from ..tasks import TrainAgentLoop
from ..diagnostics.test_installation import testInstallation
import os.path
import questionary
import sys
from ..config.logger import getLogger, setupLocalLogging
import logging


def main():
    """
        This is the entry point for the the kwola run_multiple command, which is used to run multiple consecutive Kwola runs
        one after another.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print(
            "Unable to start the training loop. There appears to be a problem "
            "with your Kwola installation or environment. Exiting.")
        exit(1)


    configDirs = sys.argv[1:]

    for configDir in configDirs:
        getLogger().info(f"Starting the training loop for {configDir}")
        config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)
        TrainAgentLoop.trainAgent(config)
        getLogger().info(f"Finished the training loop for {configDir}")

