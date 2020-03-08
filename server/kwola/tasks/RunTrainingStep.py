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
    batches = agent.prepareBatchesForTestingSequence(testSequence)

    fileName = tempfile.mktemp(".bin")

    with open(fileName, 'wb') as batchFile:
        pickle.dump(batches, batchFile)

    return fileName

@app.task
def runTrainingStep():
    environment = WebEnvironment()

    testSequences = list(TestingSequenceModel.objects(status="completed").only('id'))

    agent = DeepLearningAgent()
    agent.initialize(environment)
    agent.load()

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
                batchFilename = batchFuture.result()

                with open(batchFilename, 'rb') as batchFile:
                    batches = pickle.load(batchFile)

                os.unlink(batchFilename)

                random.shuffle(batches)
                totalReward = 0
                for batch in batches:
                    rewardLoss, tracePredictionLoss, totalLoss, batchReward = agent.learnFromBatch(batch)

                    rewardLosses.append(rewardLoss)
                    tracePredictionLosses.append(tracePredictionLoss)
                    totalLosses.append(totalLoss)
                    totalReward += batchReward

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

    environment.shutdown()

    return ""


