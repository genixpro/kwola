from mongoengine import *
import datetime
import os.path
from kwola.models.id import CustomIDField
from .utilities import saveObjectToDisk, loadObjectFromDisk


class ApplicationModel(Document):
    id = CustomIDField()

    name = StringField(required=True)

    url = StringField(required=True)

    def saveToDisk(self, config):
        saveObjectToDisk(self, "applications", config)


    @staticmethod
    def loadFromDisk(id, config):
        return loadObjectFromDisk(ApplicationModel, id, "applications", config)


