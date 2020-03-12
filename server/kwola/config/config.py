import os

import os.path

import json

config_name = "laptop"

def getKwolaUserDataDirectory(subDirName):
    """
    This returns a sub-directory within the kwola user data directory.

    It will ensure the path exists prior to returning
    """

    kwolaDirectory = os.path.join(os.environ['HOME'], ".kwola")

    if not os.path.exists(kwolaDirectory):
        os.mkdir(kwolaDirectory)

    subDirectory = os.path.join(kwolaDirectory, subDirName)
    if not os.path.exists(subDirectory):
        os.mkdir(subDirectory)

    return subDirectory




def getAgentConfiguration():
    """
        This function returns the configuration for the core machine learning model of the agent.

        :return:
    """

    modelConfig = json.load(open(f"kwola/config/agent_configs/{config_name}.json"))

    return modelConfig





def getEnvironmentConfiguration():
    """
        This function returns the configuration for the core machine learning model of the agent.

        :return:
    """

    modelConfig = json.load(open(f"kwola/config/environment_configs/{config_name}.json"))

    return modelConfig



