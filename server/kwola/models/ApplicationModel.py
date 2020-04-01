from mongoengine import *
import datetime
import os.path
from kwola.models.id import CustomIDField
import json
import gzip


class ApplicationModel(Document):
    id = CustomIDField()

    name = StringField(required=True)

    url = StringField(required=True)

    def saveToDisk(self, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("applications"), str(self.id) + ".json.gz")
        with gzip.open(fileName, 'wt') as f:
            f.write(json.dumps(json.loads(self.to_json()), indent=4))


    @staticmethod
    def loadFromDisk(id, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("applications"), str(id) + ".json.gz")
        with gzip.open(fileName, 'rt') as f:
            return ApplicationModel.from_json(f.read())

