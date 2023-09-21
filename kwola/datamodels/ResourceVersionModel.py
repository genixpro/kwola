#
#     This file is copyright 2023 Bradley Allen Arsenault & Genixpro Technologies Corporation
#     See license file in the root of the project for terms & conditions.
#


from .errors.BaseError import BaseError
from .actions.BaseAction import BaseAction
from .CustomIDField import CustomIDField
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
from .EncryptedStringField import EncryptedStringField
import json
from mongoengine import *

class ResourceVersion(Document):
    meta = {
        'indexes': [
            ('resourceId',),
        ]
    }

    id = CustomIDField()

    owner = StringField()

    applicationId = StringField()

    resourceId = StringField()

    testingRunId = StringField()

    fileHash = StringField()

    canonicalFileHash = StringField()

    creationDate = DateTimeField()

    url = EncryptedStringField()

    canonicalUrl = EncryptedStringField()

    contentType = EncryptedStringField()

    didRewriteResource = BooleanField()

    rewritePluginName = StringField()

    rewriteMode = StringField()

    rewriteMessage = StringField()

    originalLength = IntField()

    rewrittenLength = IntField()

    methods = ListField(StringField())

    def saveToDisk(self, config, overrideSaveFormat=None, overrideCompression=None):
        saveObjectToDisk(self, "resource_versions", config, overrideSaveFormat=overrideSaveFormat, overrideCompression=overrideCompression)

    @staticmethod
    def loadFromDisk(id, config, printErrorOnFailure=True):
        return loadObjectFromDisk(ResourceVersion, id, "resource_versions", config, printErrorOnFailure=printErrorOnFailure)

    def loadOriginalResourceContents(self, config):
        return config.loadKwolaFileData("resource_original_contents", self.id, useCacheBucket=True)

    def saveOriginalResourceContents(self, config, data):
        return config.saveKwolaFileData("resource_original_contents", self.id, data, useCacheBucket=True)

    def loadTranslatedResourceContents(self, config):
        return config.loadKwolaFileData("resource_translated_contents", self.id, useCacheBucket=True)

    def saveTranslatedResourceContents(self, config, data):
        return config.saveKwolaFileData("resource_translated_contents", self.id, data, useCacheBucket=True)

    def unencryptedJSON(self):
        data = json.loads(self.to_json())
        for key, fieldType in ResourceVersion.__dict__.items():
            if isinstance(fieldType, EncryptedStringField) and key in data:
                data[key] = EncryptedStringField.decrypt(data[key])
        return data
