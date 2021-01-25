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
from .DiskUtilities import saveObjectToDisk, loadObjectFromDisk
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

    creationDate = DateField()

    url = StringField()

    canonicalUrl = StringField()

    contentType = StringField()

    didRewriteResource = BooleanField()

    rewritePluginName = StringField()

    rewriteMode = StringField()

    rewriteMessage = StringField()

    originalLength = IntField()

    rewrittenLength = IntField()

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

