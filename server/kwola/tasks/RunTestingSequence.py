from .celery_config import app


@app.task
def runTestingSequence(testingSequenceId):

    print(testingSequenceId)

    return ""


