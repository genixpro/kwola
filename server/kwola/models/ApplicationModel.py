from mongoengine import *
import datetime
from kwola.config.config import getKwolaUserDataDirectory
import os.path
from kwola.models.id import CustomIDField
import json


class ApplicationModel(Document):
    id = CustomIDField()

    name = StringField(required=True)

    url = StringField(required=True)

    def saveToDisk(self):
        fileName = os.path.join(getKwolaUserDataDirectory("applications"), str(self.id) + ".json")
        with open(fileName, 'wt') as f:
            f.write(json.dumps(json.loads(self.to_json()), indent=4))


    @staticmethod
    def loadFromDisk(id):
        fileName = os.path.join(getKwolaUserDataDirectory("applications"), str(id) + ".json")
        with open(fileName, 'rt') as f:
            return ApplicationModel.from_json(f.read())

