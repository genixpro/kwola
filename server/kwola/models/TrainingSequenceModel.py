from mongoengine import *
from kwola.config.config import getKwolaUserDataDirectory
import os.path
from kwola.models.id import CustomIDField
import json



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


    def saveToDisk(self):
        fileName = os.path.join(getKwolaUserDataDirectory("training_sequences"), str(self.id) + ".json")
        with open(fileName, 'wt') as f:
            f.write(json.dumps(json.loads(self.to_json()), indent=4))


    @staticmethod
    def loadFromDisk(id):
        fileName = os.path.join(getKwolaUserDataDirectory("training_sequences"), str(id) + ".json")
        if not os.path.exists(fileName):
            return None
        with open(fileName, 'rt') as f:
            return TrainingSequence.from_json(f.read())
