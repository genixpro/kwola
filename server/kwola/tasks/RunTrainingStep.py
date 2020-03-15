from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.TrainingStepModel import TrainingStep
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.config.config import getKwolaUserDataDirectory
from kwola.components.TaskProcess import TaskProcess
import concurrent.futures
import random
import numpy
import gzip
from datetime import datetime
import traceback
import redis
import json
import bson
import tempfile
import pickle
import os
from kwola.config import config

def prepareBatchesForExecutionSession(testingSequenceId, executionSessionId, batchDirectory):
    testSequence = TestingSequenceModel.objects(id=bson.ObjectId(testingSequenceId)).first()
    executionSession = ExecutionSession.objects(id=bson.ObjectId(executionSessionId)).first()

    agent = DeepLearningAgent(agentConfiguration=config.getAgentConfiguration(), whichGpu=None)

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

    batchCache = redis.Redis(db=3)

    shuffling = random.randint(0, config.getAgentConfiguration()['training_number_shufflings_cached_per_execution_sequence'])
    cacheId = str(executionSessionId) + f"_shuffling_{shuffling}"

    cached = batchCache.get(cacheId)
    if cached is not None:
        batches = pickle.loads(cached)
        return batches

    future = processExecutor.submit(prepareBatchesForExecutionSession, testingSequenceId, str(executionSessionId), batchDirectory)

    batchFilenames = future.result()

    with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
        batches = threadExecutor.map(loadBatch, batchFilenames)

    batches = list(batches)

    batchCache.set(cacheId, pickle.dumps(batches))

    return batches


def printMovingAverageLosses(trainingStep):
    agentConfig = config.getAgentConfiguration()
    movingAverageLength = int(agentConfig['print_loss_moving_average_length'])

    averageTotalRewardLoss = numpy.mean(trainingStep.totalRewardLosses[-movingAverageLength:])
    averagePresentRewardLoss = numpy.mean(trainingStep.presentRewardLosses[-movingAverageLength:])
    averageDiscountedFutureRewardLoss = numpy.mean(trainingStep.discountedFutureRewardLosses[-movingAverageLength:])
    averageTracePredictionLoss = numpy.mean(trainingStep.tracePredictionLosses[-movingAverageLength:])
    averageExecutionFeatureLoss = numpy.mean(trainingStep.executionFeaturesLosses[-movingAverageLength:])
    averageTargetHomogenizationLoss = numpy.mean(trainingStep.targetHomogenizationLosses[-movingAverageLength:])
    averagePredictedCursorLoss = numpy.mean(trainingStep.predictedCursorLosses[-movingAverageLength:])
    averageTotalLoss = numpy.mean(trainingStep.totalLosses[-movingAverageLength:])
    averageTotalRebalancedLoss = numpy.mean(trainingStep.totalRebalancedLosses[-movingAverageLength:])

    print("Moving Average Total Reward Loss:", averageTotalRewardLoss, flush=True)
    print("Moving Average Present Reward Loss:", averagePresentRewardLoss, flush=True)
    print("Moving Average Discounted Future Reward Loss:", averageDiscountedFutureRewardLoss, flush=True)
    print("Moving Average Trace Prediction Loss:", averageTracePredictionLoss, flush=True)
    print("Moving Average Execution Feature Loss:", averageExecutionFeatureLoss, flush=True)
    print("Moving Average Target Homogenization Loss:", averageTargetHomogenizationLoss, flush=True)
    print("Moving Average Predicted Cursor Loss:", averagePredictedCursorLoss, flush=True)
    print("Moving Average Total Loss:", averageTotalLoss, flush=True)
    print("Moving Average Total Rebalanced Loss:", averageTotalRebalancedLoss, flush=True)


def runTrainingStep(trainingSequenceId):
    agentConfig = config.getAgentConfiguration()

    print("Starting Training Step", flush=True)
    testSequences = list(TestingSequenceModel.objects(status="completed").order_by('-startTime').only('status', 'startTime', 'executionSessions').limit(agentConfig['training_number_of_recent_testing_sequences_to_use']))
    if len(testSequences) == 0:
        print("Error, no test sequences in db to train on for training step.", flush=True)
        print("==== Training Step Completed ====", flush=True)
        return {}

    trainingStep = TrainingStep()
    trainingStep.startTime = datetime.now()
    trainingStep.trainingSequenceId = trainingSequenceId
    trainingStep.status = "running"
    trainingStep.numberOfIterationsCompleted = 0
    trainingStep.presentRewardLosses = []
    trainingStep.discountedFutureRewardLosses = []
    trainingStep.tracePredictionLosses = []
    trainingStep.executionFeaturesLosses = []
    trainingStep.targetHomogenizationLosses = []
    trainingStep.predictedCursorLosses = []
    trainingStep.totalRewardLosses = []
    trainingStep.totalLosses = []
    trainingStep.totalRebalancedLosses = []
    trainingStep.save()

    executionSessions = []
    for testSequence in testSequences:
        for session in testSequence.executionSessions:
            executionSessions.append((str(testSequence.id), str(session.id)))

    environment = WebEnvironment(environmentConfiguration=config.getWebEnvironmentConfiguration())

    agent = DeepLearningAgent(agentConfiguration=config.getAgentConfiguration(), whichGpu="all")
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
        totalSessionsNeeded = 1 + (agentConfig['iterations_per_training_step'] * agentConfig['batch_size']) / agentConfig['testing_sequence_length']
        sessionsPrepared = 0

        batchFutures = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=agentConfig['training_max_main_process_workers']) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=agentConfig['training_max_main_thread_workers']) as threadExecutor:
                # First we chuck some sequence requests into the queue into the queue
                for n in range(agentConfig['training_precompute_sessions_count'] + agentConfig['training_execution_sessions_in_one_loop']):
                    batchFutures.append(threadExecutor.submit(prepareAndLoadBatches, executionSessions, batchDirectory, processExecutor))
                    sessionsPrepared += 1

                while trainingStep.numberOfIterationsCompleted < agentConfig['iterations_per_training_step']:
                    batches = []

                    # Snag the batches from several different sessions and shuffle them together in memory before feeding to to the network. Reduces the effect
                    # of bias caused by training several successive iterations with the same sequence.
                    for n in range(agentConfig['training_execution_sessions_in_one_loop']):
                        # Wait on the first future in the list to finish
                        sessionBatches = batchFutures.pop(0).result()

                        if sessionsPrepared < totalSessionsNeeded + 1:
                            # Request another session be prepared
                            batchFutures.append(threadExecutor.submit(prepareAndLoadBatches, executionSessions, batchDirectory, processExecutor))

                        batches.extend(sessionBatches)

                    print(f"Training on {agentConfig['training_execution_sessions_in_one_loop']} execution sessions with ", len(batches), " batches between them.", flush=True)

                    # Randomly shuffle all the batches from the different execution sessions in with one another.
                    random.shuffle(batches)

                    for batch in batches:
                        totalReward = 0

                        totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, tracePredictionLoss, executionFeaturesLoss, targetHomogenizationLoss, predictedCursorLoss, totalLoss, totalRebalancedLoss, batchReward = agent.learnFromBatch(batch)

                        trainingStep.presentRewardLosses.append(presentRewardLoss)
                        trainingStep.discountedFutureRewardLosses.append(discountedFutureRewardLoss)
                        trainingStep.tracePredictionLosses.append(tracePredictionLoss)
                        trainingStep.executionFeaturesLosses.append(executionFeaturesLoss)
                        trainingStep.targetHomogenizationLosses.append(targetHomogenizationLoss)
                        trainingStep.predictedCursorLosses.append(predictedCursorLoss)
                        trainingStep.totalRewardLosses.append(totalRewardLoss)
                        trainingStep.totalRebalancedLosses.append(totalRebalancedLoss)
                        trainingStep.totalLosses.append(totalLoss)
                        totalReward += batchReward
                        trainingStep.numberOfIterationsCompleted += 1

                        if trainingStep.numberOfIterationsCompleted % agentConfig['print_loss_iterations'] == (agentConfig['print_loss_iterations']-1):
                            print("Completed", trainingStep.numberOfIterationsCompleted, "batches", flush=True)
                            printMovingAverageLosses(trainingStep)

                        if trainingStep.numberOfIterationsCompleted % agentConfig['iterations_between_db_saves'] == (agentConfig['iterations_between_db_saves']-1):
                            trainingStep.save()

        trainingStep.endTime = datetime.now()
        trainingStep.averageTimePerIteration = (trainingStep.endTime - trainingStep.startTime).total_seconds() / trainingStep.numberOfIterationsCompleted
        trainingStep.averageLoss = float(numpy.mean(trainingStep.totalLosses))
        trainingStep.status = "completed"
        trainingStep.save()

        agent.save()
        print("Agent saved!", flush=True)

    except Exception as e:
        print(f"Error occurred while learning sequence!", flush=True)
        traceback.print_exc()
    finally:
        files = os.listdir(batchDirectory)
        for file in files:
            os.unlink(os.path.join(batchDirectory, file))
        os.rmdir(batchDirectory)

        del agent

    # This print statement will trigger the parent manager process to kill this process.
    print("==== Training Step Completed ====", flush=True)
    return {"trainingStepId": str(trainingStep.id)}


@app.task
def runTrainingStepTask(trainingSequenceId):
    runTrainingStep(trainingSequenceId)



if __name__ == "__main__":
    task = TaskProcess(runTrainingStep)
    task.run()
