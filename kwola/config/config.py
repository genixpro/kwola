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
from google.cloud import storage
import google
import google.cloud
from ..config.logger import getLogger, setupLocalLogging
from ..components.utils.retry import autoretry

globalCachedPrebuiltConfigs = {}

class KwolaCoreConfiguration:
    """
        This class represents the configuration for the Kwola model.
    """
    def __init__(self, data):
        if isinstance(data, str):
            data = json.loads(data)
        elif isinstance(data, KwolaCoreConfiguration):
            data = data.configData

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

    @staticmethod
    def loadConfigurationFromDirectory(configurationDirectory):
        configFileName = os.path.join(configurationDirectory, "kwola.json")
        configData = {}

        maxAttempts = 5
        for attempt in range(maxAttempts):
            try:
                with open(configFileName, "rt") as f:
                    configData = json.load(f)
                    break
            except OSError:
                if attempt == (maxAttempts - 1):
                    raise
                else:
                    time.sleep(2**attempt)

        configData['configurationDirectory'] = configurationDirectory
        configData['configFileName'] = configFileName

        return KwolaCoreConfiguration(configData)

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

    def serialize(self):
        return json.dumps(self.configData, indent=4)

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

    @autoretry()
    def saveKwolaFileData(self, folder, fileName, fileData, useCacheBucket=False):
        filePath = os.path.join(folder, fileName)

        if self['data_file_storage_method'] == 'local':
            # Todo - we shouldn't be making these os.path.exists calls every single time we save file data
            # Its inefficient.
            if not os.path.exists(os.path.join(self.configurationDirectory, folder)):
                try:
                    os.mkdir(os.path.join(self.configurationDirectory, folder))
                except FileExistsError:
                    # This just means there is a race condition and multiple threads attempted
                    # to create this folder at the same time.
                    pass

            with open(os.path.join(self.configurationDirectory, filePath), 'wb') as f:
                f.write(fileData)
        elif self['data_file_storage_method'] == 'gcs':
            if 'applicationId' not in self or self.applicationId is None:
                raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

            storageClient = getSharedGCSStorageClient()
            bucketId = "kwola-testing-run-data-" + self.applicationId
            if useCacheBucket:
                bucketId += "-cache"
            applicationStorageBucket = storage.Bucket(storageClient, bucketId)
            objectBlob = storage.Blob(filePath, applicationStorageBucket)
            objectBlob.upload_from_string(fileData)
        else:
            raise RuntimeError(f"Unexpected value {self['data_file_storage_method']} for configuration data_file_storage_method")

    @autoretry()
    def loadKwolaFileData(self, folder, fileName, printErrorOnFailure=True, useCacheBucket=False):
        filePath = os.path.join(folder, fileName)

        try:
            if self['data_file_storage_method'] == 'local':
                with open(os.path.join(self.configurationDirectory, filePath), 'rb') as f:
                    data = f.read()
                    return data
            elif self['data_file_storage_method'] == 'gcs':
                if 'applicationId' not in self or self.applicationId is None:
                    raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

                storageClient = getSharedGCSStorageClient()
                bucketId = "kwola-testing-run-data-" + self.applicationId
                if useCacheBucket:
                    bucketId += "-cache"
                applicationStorageBucket = storage.Bucket(storageClient, bucketId)
                objectBlob = storage.Blob(filePath, applicationStorageBucket)
                data = objectBlob.download_as_string()
                return data
            else:
                raise RuntimeError(f"Unexpected value {self['data_file_storage_method']} for configuration data_file_storage_method")
        except FileNotFoundError:
            if printErrorOnFailure:
                getLogger().info(f"Error: Failed to load file {filePath}. File not found. Usually implies the file failed to write. "
                                      "Sometimes this occurs if you kill the process while it is running. If this occurs "
                                      "during normal operations without interruption, that would indicate a bug.")
            return
        except google.cloud.exceptions.NotFound:
            if printErrorOnFailure:
                getLogger().info(f"Error: Failed to load object {filePath}. Google cloud storage file not found. Usually implies the file failed to write. "
                                      "Sometimes this occurs if you kill the process while it is running. If this occurs "
                                      "during normal operations without interruption, that would indicate a bug.")
            return

    @autoretry()
    def deleteKwolaFileData(self, folder, fileName, useCacheBucket=False):
        filePath = os.path.join(folder, fileName)

        try:
            if self['data_file_storage_method'] == 'local':
                os.unlink(os.path.join(self.configurationDirectory, filePath))
            elif self['data_file_storage_method'] == 'gcs':
                if 'applicationId' not in self or self.applicationId is None:
                    raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

                storageClient = getSharedGCSStorageClient()
                bucketId = "kwola-testing-run-data-" + self.applicationId
                if useCacheBucket:
                    bucketId += "-cache"
                applicationStorageBucket = storage.Bucket(storageClient, bucketId)
                objectBlob = storage.Blob(filePath, applicationStorageBucket)
                objectBlob.delete()
                return
            else:
                raise RuntimeError(f"Unexpected value {self['data_file_storage_method']} for configuration data_file_storage_method")
        except FileNotFoundError:
            return
        except google.cloud.exceptions.NotFound:
            return

    @autoretry()
    def listAllFilesInFolder(self, folder, useCacheBucket=False):
        if self['data_file_storage_method'] == 'local':
            dir = os.path.join(self.configurationDirectory, folder)
            if os.path.exists(dir):
                return os.listdir(dir)
            else:
                return []
        elif self['data_file_storage_method'] == 'gcs':
            if 'applicationId' not in self or self.applicationId is None:
                raise RuntimeError("Can't load object from google cloud storage without an applicationId, which is used to indicate the bucket.")

            storageClient = getSharedGCSStorageClient()

            bucketId = "kwola-testing-run-data-" + self.applicationId
            if useCacheBucket:
                bucketId += "-cache"
            applicationStorageBucket = storage.Bucket(storageClient, bucketId)

            blobs = applicationStorageBucket.list_blobs(prefix=folder, delimiter="")

            return [blob.name[len(folder) + 1:] for blob in blobs]
        else:
            raise RuntimeError(f"Unexpected value {self['data_file_storage_method']} for configuration data_file_storage_method")


globalStorageClient = None
def getSharedGCSStorageClient():
    global globalStorageClient
    if globalStorageClient is None:
        globalStorageClient = storage.Client()

    return globalStorageClient

