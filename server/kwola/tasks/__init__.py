from .RunTestingSequence import runTestingSequence
from celery import Celery
from .celery_config import app


allTasks = [
    runTestingSequence
]

