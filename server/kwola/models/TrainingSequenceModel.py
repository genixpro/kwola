from mongoengine import *
import os.path
from kwola.models.id import CustomIDField
import json
import gzip



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
        with gzip.open(fileName, 'wt') as f:
            f.write(json.dumps(json.loads(self.to_json()), indent=4))


    @staticmethod
    def loadFromDisk(id, config):
        fileName = os.path.join(config.getKwolaUserDataDirectory("training_sequences"), str(id) + ".json.gz")
        if not os.path.exists(fileName):
            return None
        with gzip.open(fileName, 'rt') as f:
            return TrainingSequence.from_json(f.read())
