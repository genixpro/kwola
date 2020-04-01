import kwola.tasks.TrainAgentLoop
import mongoengine
import sys
from kwola.config.config import Configuration
import os.path
import re

def main():
    commandArgs = sys.argv[1:]

    cantStartMessage = """"
Error! Can not start Kwola. You must provide either a web URL or the directory name of an existing Kwola run. 
The URL must be a valid url including the http:// part. If a directory name, the directory must be accessible
from the current working folder, and must have the kwola.json configuration file contained within it.
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

            configDir = Configuration.createNewLocalKwolaConfigDir("laptop", url)
            print(f"Starting a fresh Kwola run in directory {configDir} targeting URL {url}")
        else:
            print(cantStartMessage)
            exit(2)

    kwola.tasks.TrainAgentLoop.trainAgent(configDir)


