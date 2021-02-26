#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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

