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

from .main import getConfigurationDirFromCommandLineArgs
from ..components.environments.WebEnvironment import WebEnvironment
from ..config.config import Configuration
from ..diagnostics.test_installation import testInstallation
import time

def main():
    """
        This is the entry point for for the kwola full testing sequence.
    """
    success = testInstallation(verbose=True)
    if not success:
        print(
            "Unable to start the training loop. There appears to be a problem "
            "with your Kwola installation or environment. Exiting.")
        exit(1)

    configDir = getConfigurationDirFromCommandLineArgs()

    config = Configuration(configDir)

    config['web_session_headless'] = False

    # Load up the environment
    environment = WebEnvironment(config, sessionLimit=1)

    time.sleep(5)

    environment.shutdown()
    del environment

