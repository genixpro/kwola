from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
import concurrent.futures
import random
import numpy
from datetime import datetime
import traceback
import bson
import tempfile
import pickle
import os


def prepareBatchesParallel(testingSequenceId):
    testSequence = TestingSequenceModel.objects(id=bson.ObjectId(testingSequenceId)).first()
    agent = DeepLearningAgent()

    batchFileNames = []
    for batch in agent.prepareBatchesForTestingSequence(testSequence):
        fileName = tempfile.mktemp(".bin")

        with open(fileName, 'wb') as batchFile:
            pickle.dump(batch, batchFile)

        batchFileNames.append(fileName)

    return batchFileNames

@app.task
def runTrainingStep():
    testSequences = list(TestingSequenceModel.objects(status="completed").only('id'))
    if len(testSequences) == 0:
        print("Error, no test sequences in db to train on for training step.")
        return

    environment = WebEnvironment()

    agent = DeepLearningAgent()
    agent.initialize(environment)
    agent.load()

    environment.shutdown()

    # Force it to destroy now to save memory
    del environment

    try:
        batchFutures = []

        rewardLosses = []
        tracePredictionLosses = []
        totalLosses = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            for n in range(10):
                testingSequence = random.choice(testSequences)
                batchFuture = executor.submit(prepareBatchesParallel, str(testingSequence.id))
                batchFutures.append(batchFuture)

            for batchFuture in concurrent.futures.as_completed(batchFutures):
                batchFilenames = batchFuture.result()
                random.shuffle(batchFilenames)

                totalReward = 0
                for batchFilename in batchFilenames:
                    with open(batchFilename, 'rb') as batchFile:
                        batch = pickle.load(batchFile)

                    os.unlink(batchFilename)

                    rewardLoss, tracePredictionLoss, totalLoss, batchReward = agent.learnFromBatch(batch)

                    rewardLosses.append(rewardLoss)
                    tracePredictionLosses.append(tracePredictionLoss)
                    totalLosses.append(totalLoss)
                    totalReward += batchReward

                print("Completed", len(batchFilenames), "batches")

        averageRewardLoss = numpy.mean(rewardLosses)
        averageTracePredictionLoss = numpy.mean(tracePredictionLosses)
        averageTotalLoss = numpy.mean(totalLosses)

        # print(testingSequence.id)
        # print("Total Reward", float(totalReward))
        print("Average Reward Loss:", averageRewardLoss)
        print("Average Trace Predicton Loss:", averageTracePredictionLoss)
        print("Average Total Loss:", averageTotalLoss)

    except Exception as e:
        print(f"Error occurred while learning sequence!")
        traceback.print_exc()

    agent.save()

    return ""


