from mongoengine import *
import os.path
from kwola.models.id import CustomIDField
import json
import gzip
from .lockedfile import LockedFile



class TrainingSequence(Document):
    id = CustomIDField()

    applicationId = StringField()

    status = StringField()

    startTime = DateTimeField()

    endTime = DateTimeField()

    trainingStepsCompleted = IntField()

    initializationTestingSteps = ListField(StringField())

    testingSteps = ListField(StringField())

    trainingSteps = ListField(StringField())

    averageTimePerStep = FloatField()


    def saveToDisk(self, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("training_sequences"), str(self.id) + ".json.gz")
        with LockedFile(fileName, 'wb') as f:
            f.write(gzip.compress(bytes(json.dumps(json.loads(self.to_json()), indent=4), "utf8")))


    @staticmethod
    def loadFromDisk(id, config):
        try:
            fileName = os.path.join(config.getKwolaUserDataDirectory("training_sequences"), str(id) + ".json.gz")
            if not os.path.exists(fileName):
                return None
            with LockedFile(fileName, 'rb') as f:
                return TrainingSequence.from_json(str(gzip.decompress(f.read()), "utf8"))
        except json.JSONDecodeError:
            return



