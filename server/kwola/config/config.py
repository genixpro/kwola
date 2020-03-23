import os

import os.path

import json

config_name = "rig"

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



globalAgentConfiguration = None
def getAgentConfiguration():
    """
        This function returns the configuration for the core machine learning model of the agent.

        :return:
    """
    global globalAgentConfiguration

    if globalAgentConfiguration is not None:
        return globalAgentConfiguration

    globalAgentConfiguration = json.load(open(f"kwola/config/agent_configs/{config_name}.json"))

    return globalAgentConfiguration





globalWebEnvironmentConfiguration = None
def getWebEnvironmentConfiguration():
    """
        This function returns the configuration for the core machine learning model of the agent.

        :return:
    """
    global globalWebEnvironmentConfiguration

    if globalWebEnvironmentConfiguration is not None:
        return globalWebEnvironmentConfiguration

    globalWebEnvironmentConfiguration = json.load(open(f"kwola/config/web_environment_configs/{config_name}.json"))

    return globalWebEnvironmentConfiguration



