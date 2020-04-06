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


from ..tasks import TrainAgentLoop
import mongoengine
import sys
from ..config.config import Configuration
import os.path
import re
import questionary

def main():
    commandArgs = sys.argv[1:]

    cantStartMessage = """"
Error! Can not start .. You must provide either a web URL or the directory name of an existing Kwola run. 
The URL must be a valid url including the http:// part. If a directory name, the directory must be accessible
from the current working folder, and must have the ..json configuration file contained within it.
Please try again.
    """

    configDir = None

    if len(commandArgs) == 0:
        configDir = Configuration.findLocalKwolaConfigDirectory()
        if configDir is None:
            print(cantStartMessage)
            exit(1)
        else:
            print(f"Restarting the Kwola run in directory {configDir}")

    elif len(commandArgs) == 1:
        secondArg = commandArgs[0]

        if os.path.exists(secondArg) and Configuration.checkDirectoryContainsKwolaConfig(secondArg):
            configDir = secondArg
            print(f"Restarting the Kwola run in directory {configDir}")
        elif Configuration.isValidURL(secondArg):
            # Create a new config directory for this URL
            url = secondArg

            configName = questionary.select(
                "Which configuration do you want to load for your model?",
                choices=[
                    'testing',
                    'small',
                    'medium',
                    'large'
                ]).ask()  # returns value of selection

            configDir = Configuration.createNewLocalKwolaConfigDir(configName, url)
            print(f"Starting a fresh Kwola run in directory {configDir} targeting URL {url}")
        else:
            print(cantStartMessage)
            exit(2)

    TrainAgentLoop.trainAgent(configDir)


