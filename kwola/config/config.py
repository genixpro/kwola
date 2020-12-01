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


import json
import os
import os.path
import pkg_resources
import re
import time
import pymongo.errors
import mongoengine
import mongoengine.connection
from pprint import pprint
from ..components.utils.regex import sharedUrlRegex


globalCachedPrebuiltConfigs = {}

class KwolaCoreConfiguration:
    """
        This class represents the configuration for the Kwola model.
    """
    def __init__(self, configurationDirectory = None, configData = None):
        if configurationDirectory is not None:
            self.configFileName = os.path.join(configurationDirectory, "kwola.json")
            self.configurationDirectory = configurationDirectory
            self.configData = {}

            maxAttempts = 5
            for attempt in range(maxAttempts):
                try:
                    with open(self.configFileName, "rt") as f:
                        data = json.load(f)
                        break
                except OSError:
                    if attempt == (maxAttempts - 1):
                        raise
                    else:
                        time.sleep(2**attempt)
        else:
            data = configData


        if 'profile' not in data:
            data['profile'] = 'medium'

        try:
            # If there are any configuration values that exist in the prebuilt config
            # that we don't see in the loaded configuration file, then we add in
            # those keys with the default values taken from the prebuilt config.
            # This allows people to continue running existing runs even if we add
            # new configuration keys in subsequent releases.
            prebuiltConfigData = KwolaCoreConfiguration.getPrebuiltConfigData(data['profile'])
            for key, value in prebuiltConfigData.items():
                if key not in data:
                    data[key] = value

        except FileNotFoundError:
            # This indicates the prebuilt configuration could not be found.
            # Print an error message and ignore it.
            print(f"Was unable to find the prebuilt configuration file for {data['profile']}. Skipping loading default values.")

        self.configData = data

        self.connectToMongoIfNeeded()

    def connectToMongoIfNeeded(self):
        if isinstance(self.configData['data_serialization_method'], dict):
            method = self.configData['data_serialization_method']['default']
        else:
            method = self.configData['data_serialization_method']

        if method == "mongo" and 'mongo_uri' in self.configData and self.configData['mongo_uri'] and len(mongoengine.connection._connections) == 0:
            maxAttempts = 5
            for attempt in range(maxAttempts):
                try:
                    mongoengine.connect(host=self.configData['mongo_uri'])
                    break
                except Exception as e:
                    if attempt == (maxAttempts - 1):
                        raise
                    else:
                        time.sleep(2**attempt)


    def getKwolaUserDataDirectory(self, subDirName, ensureExists=True):
        """
        This returns a sub-directory within the kwola user data directory.

        ensureExists - when set to True (default), this will ensure the directory exists first before returning it
        """

        if ensureExists:
            if not os.path.exists(self.configurationDirectory):
                os.mkdir(self.configurationDirectory)

        subDirectory = os.path.join(self.configurationDirectory, subDirName)
        if ensureExists:
            if not os.path.exists(subDirectory):
                try:
                    os.mkdir(subDirectory)
                except FileExistsError:
                    pass

        return subDirectory

    def __contains__(self, key):
        return key in self.configData

    def __getitem__(self, key):
        try:
            return self.configData[key]
        except KeyError:
            pprint(self.configData)
            raise

    def __setitem__(self, key, value):
        self.configData[key] = value

    def __getattr__(self, name):
        if name != "configData" and name in self.configData:
            return self.configData[name]
        else:
            # Default behaviour
            raise AttributeError()

    def saveConfig(self):
        with open(self.configFileName, "wt") as f:
            json.dump(self.configData, f, indent=4, sort_keys=True)

    @staticmethod
    def checkDirectoryContainsKwolaConfig(directory):
        if os.path.exists(os.path.join(directory, "kwola.json")):
            return True
        else:
            return False


    @staticmethod
    def findLocalKwolaConfigDirectory():
        currentDir = os.getcwd()

        found = []
        subFilesFolders = os.listdir(currentDir)
        for subDir in subFilesFolders:
            if KwolaCoreConfiguration.checkDirectoryContainsKwolaConfig(subDir):
                found.append(subDir)

        found = sorted(found, reverse=True)

        return found[0]

    @staticmethod
    def createNewLocalKwolaConfigDir(prebuild, **configArgs):
        n = 1
        while True:
            dirname = f"kwola_run_{n:03d}"
            if not os.path.exists(dirname):
                os.mkdir(dirname)

                with open(os.path.join(dirname, "kwola.json"), "wt") as configFile:
                    prebuildConfigData = KwolaCoreConfiguration.getPrebuiltConfigData(prebuild)
                    for key, value in configArgs.items():
                        prebuildConfigData[key] = value
                    configFile.write(json.dumps(prebuildConfigData, indent=4))

                return dirname
            else:
                n += 1


    @staticmethod
    def isValidURL(url):
        return re.match(sharedUrlRegex, url) is not None

    @staticmethod
    def getPrebuiltConfigData(prebuild):
        global globalCachedPrebuiltConfigs

        if prebuild in globalCachedPrebuiltConfigs:
            return globalCachedPrebuiltConfigs[prebuild]

        localFilePath = f"{prebuild}.json"
        if os.path.exists(localFilePath):
            with open(localFilePath, 'rt') as f:
                data = json.load(f)
                globalCachedPrebuiltConfigs[prebuild] = data
                return data
        else:
            data = json.loads(pkg_resources.resource_string("kwola", os.path.join("config", "prebuilt_configs", f"{prebuild}.json")))
            globalCachedPrebuiltConfigs[prebuild] = data
            return data


