#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#

from .main import getConfigurationDirFromCommandLineArgs
from ..components.environments.WebEnvironment import WebEnvironment
from ..config.config import KwolaCoreConfiguration
from ..diagnostics.test_installation import testInstallation
import time
from ..config.logger import getLogger, setupLocalLogging
import logging

def main():
    """
        This is the entry point for for the kwola full testing sequence.
    """
    setupLocalLogging()
    success = testInstallation(verbose=True)
    if not success:
        print(
            "Unable to start the training loop. There appears to be a problem "
            "with your Kwola installation or environment. Exiting.")
        exit(1)

    configDir = getConfigurationDirFromCommandLineArgs()

    config = KwolaCoreConfiguration.loadConfigurationFromDirectory(configDir)

    config['web_session_headless'] = False

    # Load up the environment
    environment = WebEnvironment(config, sessionLimit=1)

    time.sleep(5)

    environment.shutdown()
    del environment

