from mongoengine import *



class TrainingSequence(Document):
    applicationId = StringField()

    status = StringField()

    startTime = DateTimeField()

    endTime = DateTimeField()

    trainingStepsCompleted = IntField()

    initializationTestingSequences = ListField(GenericReferenceField())

    testingSequences = ListField(GenericReferenceField())

    trainingSteps = ListField(GenericReferenceField())

    averageTimePerStep = FloatField()

