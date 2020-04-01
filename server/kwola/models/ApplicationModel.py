from mongoengine import *
import datetime
import os.path
from kwola.models.id import CustomIDField
import json
import gzip
from .lockedfile import LockedFile


class ApplicationModel(Document):
    id = CustomIDField()

    name = StringField(required=True)

    url = StringField(required=True)

    def saveToDisk(self, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("applications"), str(self.id) + ".json.gz")
        with LockedFile(fileName, 'wb') as f:
            f.write(gzip.compress(bytes(json.dumps(json.loads(self.to_json()), indent=4), "utf8")))


    @staticmethod
    def loadFromDisk(id, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("applications"), str(id) + ".json.gz")
        with LockedFile(fileName, 'rb') as f:
            return ApplicationModel.from_json(str(gzip.decompress(f.read()), "utf8"))


