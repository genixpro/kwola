from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.config.config import getKwolaUserDataDirectory
import concurrent.futures
import random
import numpy
import gzip
from datetime import datetime
import traceback
import bson
import tempfile
import pickle
import os

def prepareBatchesForExecutionSession(testingSequenceId, executionSessionId, batchDirectory):
    testSequence = TestingSequenceModel.objects(id=bson.ObjectId(testingSequenceId)).first()
    executionSession = ExecutionSession.objects(id=bson.ObjectId(executionSessionId)).first()

    agent = DeepLearningAgent()

    sequenceBatches = agent.prepareBatchesForExecutionSession(testSequence, executionSession)

    fileNames = []

    for batch in sequenceBatches:
        fileDescriptor, fileName = tempfile.mkstemp(".bin", dir=batchDirectory)

        with open(fileDescriptor, 'wb') as batchFile:
            # compressedBatchFile = gzip.open(batchFile, 'wb', compresslevel=5)
            pickle.dump(batch, batchFile)
            # compressedBatchFile.close()

        fileNames.append(fileName)

    # print("Finished writing all batches for execution session", executionSessionId)

    return fileNames


def loadBatch(batchFilename):
    with open(batchFilename, 'rb') as batchFile:
        # compressedBatchFile = gzip.open(batchFile, 'rb')
        batch = pickle.load(batchFile)
        # compressedBatchFile.close()

    os.unlink(batchFilename)

    return batch

def prepareAndLoadBatches(executionSessions, batchDirectory, processExecutor):
    testingSequenceId, executionSessionId = random.choice(executionSessions)
    future = processExecutor.submit(prepareBatchesForExecutionSession, testingSequenceId, str(executionSessionId), batchDirectory)

    batchFilenames = future.result()

    with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
        batches = threadExecutor.map(loadBatch, batchFilenames)

    return list(batches)


@app.task
def runTrainingStep():
    print("Starting Training Step")
    testSequences = list(TestingSequenceModel.objects(status="completed").only('executionSessions'))
    if len(testSequences) == 0:
        print("Error, no test sequences in db to train on for training step.")
        return

    executionSessions = []
    for testSequence in testSequences:
        for session in testSequence.executionSessions:
            executionSessions.append((str(testSequence.id), str(session.id)))

    environment = WebEnvironment(numberParallelSessions=1)

    agent = DeepLearningAgent()
    agent.initialize(environment)
    agent.load()

    environment.shutdown()

    # Haven't decided yet whether we should force Kwola to always write to disc or spool in memory
    # using /tmp. The following lines switch between the two approaches
    # batchDirectory = tempfile.mkdtemp(dir=getKwolaUserDataDirectory("batches"))
    batchDirectory = tempfile.mkdtemp()

    # Force it to destroy now to save memory
    del environment

    try:
        batchFutures = []
        rewardLosses = []
        tracePredictionLosses = []
        totalLosses = []
        iterationsCompleted = 0
        iterationsNeeded = 500
        precomputeExecutionSessionsCount = 15 # Note each execution session produces multiple batches and thus multiple iterations

        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as threadExecutor:
                # First we chuck some sequence requests into the queue into the queue
                for n in range(precomputeExecutionSessionsCount):
                    batchFutures.append(threadExecutor.submit(prepareAndLoadBatches, executionSessions, batchDirectory, processExecutor))

                while iterationsCompleted < iterationsNeeded:
                    # Wait on the first future in the list to finish
                    batches = batchFutures.pop(0).result()

                    # Request another session be prepared
                    batchFutures.append(threadExecutor.submit(prepareAndLoadBatches, executionSessions, batchDirectory, processExecutor))

                    print("Training on execution session with ", len(batches), " batches")

                    for batch in batches:
                        totalReward = 0

                        rewardLoss, tracePredictionLoss, totalLoss, batchReward = agent.learnFromBatch(batch)

                        rewardLosses.append(rewardLoss)
                        tracePredictionLosses.append(tracePredictionLoss)
                        totalLosses.append(totalLoss)
                        totalReward += batchReward
                        iterationsCompleted += 1

                        print("Completed", iterationsCompleted, "batches")

                    averageRewardLoss = numpy.mean(rewardLosses[-25:])
                    averageTracePredictionLoss = numpy.mean(tracePredictionLosses[-25:])
                    averageTotalLoss = numpy.mean(totalLosses[-25:])

                    # print(testingSequence.id)
                    # print("Total Reward", float(totalReward))
                    print("Moving Average Reward Loss:", averageRewardLoss)
                    print("Moving Average Trace Predicton Loss:", averageTracePredictionLoss)
                    print("Moving Average Total Loss:", averageTotalLoss)

        agent.save()
        print("Agent saved!")

    except Exception as e:
        print(f"Error occurred while learning sequence!")
        traceback.print_exc()
    finally:
        files = os.listdir(batchDirectory)
        for file in files:
            os.unlink(os.path.join(batchDirectory, file))
        os.rmdir(batchDirectory)

        del environment, agent

    print("Training Step Completed")
    return ""


