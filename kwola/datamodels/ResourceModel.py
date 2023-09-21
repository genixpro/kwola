#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .errors.BaseError import BaseError
from .actions.BaseAction import BaseAction
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk, getDataFormatAndCompressionForClass
from mongoengine import *
from .EncryptedStringField import EncryptedStringField
import json
import os

class Resource(Document):
    meta = {
        'indexes': [
            ('owner', 'applicationId',),
            ('applicationId', "canonicalUrl"),
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    url = EncryptedStringField()

    canonicalUrl = EncryptedStringField()

    creationDate = DateTimeField()

    didRewriteResource = BooleanField()

    rewritePluginName = StringField()

    rewriteMode = StringField()

    rewriteMessage = StringField()

    contentType = StringField()

    versionSaveMode = StringField()

    latestVersionId = StringField()

    methods = ListField(StringField())


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
                    resourceID = resourceID.replace(".enc", "")

                    resourceIds.add(resourceID)

            objects = [
                Resource.loadFromDisk(resourceId, config)
                for resourceId in resourceIds
            ]

            return objects

    def getVersionId(self, fileHash):
        return self.id + "-" + fileHash


    def unencryptedJSON(self):
        data = json.loads(self.to_json())
        for key, fieldType in Resource.__dict__.items():
            if isinstance(fieldType, EncryptedStringField) and key in data:
                data[key] = EncryptedStringField.decrypt(data[key])
        return data
