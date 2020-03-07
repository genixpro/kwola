from .RunTestingSequence import runTestingSequence
from .RunTrainingStep import runTrainingStep
from .TrainAgent import trainAgent
from celery import Celery
from .celery_config import app
from mongoengine import connect
from multiprocessing import set_start_method

connect('kwola')

try:
    set_start_method('spawn')
except RuntimeError:
    pass

allTasks = [
    runTestingSequence,
    runTrainingStep,
    trainAgent,
]

