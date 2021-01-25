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


from .errors.BaseError import BaseError
from .actions.BaseAction import BaseAction
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk, getDataFormatAndCompressionForClass
from mongoengine import *
import os

class Resource(Document):
    meta = {
        'indexes': [
            ('owner', 'applicationId',),
            ('applicationId',),
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    url = StringField()

    canonicalUrl = StringField()

    creationDate = DateField()

    didRewriteResource = BooleanField()

    rewritePluginName = StringField()

    rewriteMode = StringField()

    rewriteMessage = StringField()

    contentType = StringField()

    versionSaveMode = StringField()

    latestVersionId = StringField()


    def saveToDisk(self, config, overrideSaveFormat=None, overrideCompression=None):
        saveObjectToDisk(self, "resources", config, overrideSaveFormat=overrideSaveFormat, overrideCompression=overrideCompression)

    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(Resource, id, "resources", config, printErrorOnFailure=printErrorOnFailure)

    @staticmethod
    def loadAllResources(config, applicationId=None):
        dataFormat, compression = getDataFormatAndCompressionForClass(Resource, config)

        if dataFormat == "mongo":
            if applicationId is None and 'applicationId' in config:
                applicationId = config['applicationId']
            objects = Resource.objects(applicationId=applicationId)
            return objects
        else:
            resourceFiles = config.listAllFilesInFolder("resources")

            resourceIds = set()

            for fileName in resourceFiles:
                if ".lock" not in fileName:
                    resourceID = fileName
                    resourceID = resourceID.replace(".json", "")
                    resourceID = resourceID.replace(".gz", "")
                    resourceID = resourceID.replace(".pickle", "")

                    resourceIds.add(resourceID)

            objects = [
                Resource.loadFromDisk(resourceId, config)
                for resourceId in resourceIds
            ]

            return objects

    def getVersionId(self, fileHash):
        return self.id + "-" + fileHash
