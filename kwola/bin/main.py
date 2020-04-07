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


from ..config.config import Configuration
from ..tasks import TrainAgentLoop
import os.path
import questionary
import sys


def getConfigurationDirFromCommandLineArgs():
    """
        This function is responsible for parsing the command line arguments and returning a directory containing a
        Kwola configuration. If none exists, a new Kwola configuration will be created.

        :return: A string containing the directory name with the configuration.
    """

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
            print(f"Loading the Kwola run in directory {configDir}")

    elif len(commandArgs) == 1:
        secondArg = commandArgs[0]

        if os.path.exists(secondArg) and Configuration.checkDirectoryContainsKwolaConfig(secondArg):
            configDir = secondArg
            print(f"Loading the Kwola run in directory {configDir}")
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

            email = questionary.text("What is the email/username to use for testing (blank disables this action)?").ask()
            password = questionary.text("What is the password to use for testing (blank disables this action)?").ask()
            name = questionary.text("What human name / short text to use for testing (blank disables this action)?").ask()
            paragraph = questionary.text("What is paragraph / long text to use (blank disables this action)?").ask()

            commandChoices = [
                "Enable random number command?",
                "Enable random bracket command?",
                "Enable random math symbol command?",
                "Enable random other symbol command?",
                "Enable double click command?",
                "Enable right click command?"
            ]

            results = questionary.checkbox("Please select which commands you want to enable", choices=commandChoices).ask()

            enableRandomNumberCommand = bool(commandChoices[0] in results)
            enableRandomBracketCommand = bool(commandChoices[1] in results)
            enableRandomMathCommand = bool(commandChoices[2] in results)
            enableRandomOtherSymbolCommand = bool(commandChoices[3] in results)
            enableDoubleClickCommand = bool(commandChoices[4] in results)
            enableRightClickCommand = bool(commandChoices[5] in results)

            configDir = Configuration.createNewLocalKwolaConfigDir(configName,
                                                                   url=url,
                                                                   email=email,
                                                                   password=password,
                                                                   name=name,
                                                                   paragraph=paragraph,
                                                                   enableRandomNumberCommand=enableRandomNumberCommand,
                                                                   enableRandomBracketCommand=enableRandomBracketCommand,
                                                                   enableRandomMathCommand=enableRandomMathCommand,
                                                                   enableRandomOtherSymbolCommand=enableRandomOtherSymbolCommand,
                                                                   enableDoubleClickCommand=enableDoubleClickCommand,
                                                                   enableRightClickCommand=enableRightClickCommand
                                                                   )

            ready = questionary.select(
                "Are you ready to unleash the Kwolas?",
                choices=[
                    'yes',
                    'no'
                ]).ask()  # returns value of selection

            if ready == "no":
                exit(0)

            print(f"Starting a fresh Kwola run in directory {configDir} targeting URL {url}")
        else:
            print(cantStartMessage)
            exit(2)

    return configDir


def main():
    """
        This is the entry point for the main Kwola application, the console command "kwola".
        All it does is start a training loop.
    """
    configDir = getConfigurationDirFromCommandLineArgs()
    TrainAgentLoop.trainAgent(configDir)
