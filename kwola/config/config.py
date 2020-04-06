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


import os
import os.path
import json
import pkg_resources
import re

class Configuration:
    """
        This class represents the configuration for the Kwola model.
    """
    def __init__(self, configurationDirectory):
        self.configFileName = os.path.join(configurationDirectory, "kwola.json")
        self.configurationDirectory = configurationDirectory

        data = json.load(open(self.configFileName, "rt"))

        for key, value in data.items():
            setattr(self, key, value)


    def getKwolaUserDataDirectory(self, subDirName):
        """
        This returns a sub-directory within the kwola user data directory.

        It will ensure the path exists prior to returning
        """

        subDirectory = os.path.join(self.configurationDirectory, subDirName)
        if not os.path.exists(subDirectory):
            try:
                os.mkdir(subDirectory)
            except FileExistsError:
                pass

        return subDirectory


    def __getitem__(self, key):
        return getattr(self, key)


    @staticmethod
    def checkDirectoryContainsKwolaConfig(directory):
        if os.path.exists(os.path.join(directory, "kwola.json")):
            return True
        else:
            return False


    @staticmethod
    def findLocalKwolaConfigDirectory():
        currentDir = os.getcwd()

        found = None
        subFilesFolders = os.listdir(currentDir)
        for subDir in subFilesFolders:
            if Configuration.checkDirectoryContainsKwolaConfig(subDir):
                found = subDir
                break

        return found

    @staticmethod
    def createNewLocalKwolaConfigDir(prebuild, url):
        n = 0
        while True:
            dirname = f"kwola_results_{n}"
            if not os.path.exists(dirname):
                os.mkdir(dirname)

                with open(os.path.join(dirname, "kwola.json"), "wt") as configFile:
                    prebuildConfigData = json.loads(pkg_resources.resource_string("kwola", f"config/prebuilt_configs/{prebuild}.json"))
                    prebuildConfigData['url'] = url
                    configFile.write(json.dumps(prebuildConfigData, indent=4))

                return dirname
            else:
                n += 1


    @staticmethod
    def isValidURL(url):
        regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return re.match(regex, url) is not None

