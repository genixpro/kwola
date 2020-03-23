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
import sklearn.preprocessing
import pickle
import os
from kwola.config import config

def prepareBatchesForExecutionSession(executionSessionId, batchDirectory):
    agentConfiguration = config.getAgentConfiguration()

    batchCache = redis.Redis(db=3)
    if agentConfiguration['training_number_shufflings_cached_per_execution_session'] > 1:
        shuffling = random.randint(0, agentConfiguration['training_number_shufflings_cached_per_execution_session'] - 1)
        cacheId = str(executionSessionId) + f"_shuffling_{shuffling}"
    else:
        cacheId = str(executionSessionId) + f"_shuffling_0"

    cached = batchCache.get(cacheId)
    if cached is not None:
        cacheHit = True

        sequenceBatches = pickle.loads(gzip.decompress(cached))
    else:
        cacheHit = False

        with open(os.path.join(config.getKwolaUserDataDirectory("models"), "reward_normalizer"), "rb") as normalizerFile:
            trainingRewardNormalizer = pickle.load(normalizerFile)

        executionSession = ExecutionSession.objects(id=bson.ObjectId(executionSessionId)).first()

        agent = DeepLearningAgent(agentConfiguration=agentConfiguration, whichGpu=None)

        sequenceBatches = agent.prepareBatchesForExecutionSession(executionSession, trainingRewardNormalizer=trainingRewardNormalizer)

        pickleBytes = pickle.dumps(sequenceBatches)
        compressedPickleBytes = gzip.compress(pickleBytes)

        # Add small random number here to ensure there isn't a bunch of entries expiring in the cache at the same time
        batchCache.set(cacheId, compressedPickleBytes, ex=agentConfiguration['training_execution_session_cache_expiration_seconds'] + random.randint(0, agentConfiguration['training_execution_session_cache_expiration_max_additional_random_seconds']))

    fileNames = []

    for batch in sequenceBatches:
        fileDescriptor, fileName = tempfile.mkstemp(".bin", dir=batchDirectory)

        with open(fileDescriptor, 'wb') as batchFile:
            pickle.dump(batch, batchFile)

        fileNames.append(fileName)

    # print(datetime.now(), "Finished writing all batches for execution session", executionSessionId)

    return fileNames, cacheHit


def loadBatch(batchFilename):
    with open(batchFilename, 'rb') as batchFile:
        # compressedBatchFile = gzip.open(batchFile, 'rb')
        batch = pickle.load(batchFile)
        # compressedBatchFile.close()

    os.unlink(batchFilename)

    return batch

def prepareAndLoadBatches(executionSessions, batchDirectory, processExecutor):
    executionSessionId = random.choice(executionSessions)

    future = processExecutor.submit(prepareBatchesForExecutionSession, str(executionSessionId), batchDirectory)

    batchFilenames, cacheHit = future.result()

    with concurrent.futures.ThreadPoolExecutor() as threadExecutor:
        batches = threadExecutor.map(loadBatch, batchFilenames)

    batches = list(batches)

    return batches, cacheHit


def printMovingAverageLosses(trainingStep):
    agentConfig = config.getAgentConfiguration()
    movingAverageLength = int(agentConfig['print_loss_moving_average_length'])

    averageStart = max(0, min(len(trainingStep.totalRewardLosses) - 1, movingAverageLength))

    averageTotalRewardLoss = numpy.mean(trainingStep.totalRewardLosses[-averageStart:])
    averagePresentRewardLoss = numpy.mean(trainingStep.presentRewardLosses[-averageStart:])
    averageDiscountedFutureRewardLoss = numpy.mean(trainingStep.discountedFutureRewardLosses[-averageStart:])
    averageTracePredictionLoss = numpy.mean(trainingStep.tracePredictionLosses[-averageStart:])
    averageExecutionFeatureLoss = numpy.mean(trainingStep.executionFeaturesLosses[-averageStart:])
    averageTargetHomogenizationLoss = numpy.mean(trainingStep.targetHomogenizationLosses[-averageStart:])
    averagePredictedCursorLoss = numpy.mean(trainingStep.predictedCursorLosses[-averageStart:])
    averageTotalLoss = numpy.mean(trainingStep.totalLosses[-averageStart:])
    averageTotalRebalancedLoss = numpy.mean(trainingStep.totalRebalancedLosses[-averageStart:])

    print(datetime.now(), "Moving Average Total Reward Loss:", averageTotalRewardLoss, flush=True)
    print(datetime.now(), "Moving Average Present Reward Loss:", averagePresentRewardLoss, flush=True)
    print(datetime.now(), "Moving Average Discounted Future Reward Loss:", averageDiscountedFutureRewardLoss, flush=True)
    print(datetime.now(), "Moving Average Trace Prediction Loss:", averageTracePredictionLoss, flush=True)
    print(datetime.now(), "Moving Average Execution Feature Loss:", averageExecutionFeatureLoss, flush=True)
    print(datetime.now(), "Moving Average Target Homogenization Loss:", averageTargetHomogenizationLoss, flush=True)
    print(datetime.now(), "Moving Average Predicted Cursor Loss:", averagePredictedCursorLoss, flush=True)
    if agentConfig['enable_loss_balancing']:
        print(datetime.now(), "Moving Average Total Raw Loss:", averageTotalLoss, flush=True)
        print(datetime.now(), "Moving Average Total Rebalanced Loss:", averageTotalRebalancedLoss, flush=True)
    else:
        print(datetime.now(), "Moving Average Total Loss:", averageTotalLoss, flush=True)


def runTrainingStep(trainingSequenceId):
    try:
        agentConfig = config.getAgentConfiguration()

        print(datetime.now(), "Starting Training Step", flush=True)
        testSequences = list(TestingSequenceModel.objects(status="completed").order_by('-startTime').only('status', 'startTime', 'executionSessions').limit(agentConfig['training_number_of_recent_testing_sequences_to_use']))
        if len(testSequences) == 0:
            print(datetime.now(), "Error, no test sequences in db to train on for training step.", flush=True)
            print(datetime.now(), "==== Training Step Completed ====", flush=True)
            return {}

        executionSessionIds = []
        for testSequence in testSequences:
            for session in testSequence.executionSessions:
                executionSessionIds.append(str(session.id))

        trainingRewardNormalizer = DeepLearningAgent.createTrainingRewardNormalizer(random.sample(executionSessionIds, min(len(executionSessionIds), agentConfig['training_reward_normalizer_fit_population_size'])))

        with open(os.path.join(config.getKwolaUserDataDirectory("models"), "reward_normalizer"), "wb") as normalizerFile:
            pickle.dump(trainingRewardNormalizer, normalizerFile)

        del testSequences

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

        environment = WebEnvironment(environmentConfiguration=config.getWebEnvironmentConfiguration())

        agent = DeepLearningAgent(agentConfiguration=config.getAgentConfiguration(), whichGpu="all")
        agent.initialize(environment.branchFeatureSize())
        agent.load()

        environment.shutdown()

        # Haven't decided yet whether we should force Kwola to always write to disc or spool in memory
        # using /tmp. The following lines switch between the two approaches
        # batchDirectory = tempfile.mkdtemp(dir=getKwolaUserDataDirectory("batches"))
        batchDirectory = tempfile.mkdtemp()

        # Force it to destroy now to save memory
        del environment

    except Exception as e:
        print(datetime.now(), f"Error occurred during initiation of training!", flush=True)
        traceback.print_exc()
        return {}

    try:
        totalSessionsNeeded = 1 + (agentConfig['iterations_per_training_step'] * agentConfig['batch_size']) / agentConfig['testing_sequence_length']
        sessionsPrepared = 0

        batchFutures = []

        recentCacheHits = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=agentConfig['training_max_main_process_workers']) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=agentConfig['training_max_main_thread_workers']) as threadExecutor:
                # First we chuck some sequence requests into the queue into the queue
                for n in range(agentConfig['training_precompute_sessions_count'] + agentConfig['training_execution_sessions_in_one_loop']):
                    batchFutures.append(threadExecutor.submit(prepareAndLoadBatches, executionSessionIds, batchDirectory, processExecutor))
                    sessionsPrepared += 1

                while trainingStep.numberOfIterationsCompleted < agentConfig['iterations_per_training_step']:
                    batches = []

                    # Snag the batches from several different sessions and shuffle them together in memory before feeding to to the network. Reduces the effect
                    # of bias caused by training several successive iterations with the same sequence.
                    for n in range(agentConfig['training_execution_sessions_in_one_loop']):
                        # Wait on the first future in the list to finish
                        sessionBatches, cacheHit = batchFutures.pop(0).result()
                        recentCacheHits.append(int(cacheHit))

                        if sessionsPrepared < totalSessionsNeeded + 1:
                            # Request another session be prepared
                            batchFutures.append(threadExecutor.submit(prepareAndLoadBatches, executionSessionIds, batchDirectory, processExecutor))

                        batches.extend(sessionBatches)

                    print(datetime.now(), f"Training on {agentConfig['training_execution_sessions_in_one_loop']} execution sessions with ", len(batches), " batches between them.", flush=True)

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
                            print(datetime.now(), "Completed", trainingStep.numberOfIterationsCompleted + 1, "batches", flush=True)
                            printMovingAverageLosses(trainingStep)
                            if agentConfig['print_cache_hit_rate']:
                                print(datetime.now(), f"Batch cache hit rate {100 * numpy.mean(recentCacheHits[-agentConfig['print_cache_hit_rate_moving_average_length']:]):.0f}%")

                        if trainingStep.numberOfIterationsCompleted % agentConfig['iterations_between_db_saves'] == (agentConfig['iterations_between_db_saves']-1):
                            trainingStep.save()

        trainingStep.endTime = datetime.now()
        trainingStep.averageTimePerIteration = (trainingStep.endTime - trainingStep.startTime).total_seconds() / trainingStep.numberOfIterationsCompleted
        trainingStep.averageLoss = float(numpy.mean(trainingStep.totalLosses))
        trainingStep.status = "completed"
        trainingStep.save()

        # Safe guard, don't save the model if any nan's were detected
        if numpy.count_nonzero(numpy.isnan(trainingStep.totalLosses)) == 0:
            agent.save()
            print(datetime.now(), "Agent saved!", flush=True)
        else:
            print("ERROR! A NaN was detected in this models output. Not saving model.", flush=True)

    except Exception as e:
        print(datetime.now(), f"Error occurred while learning sequence!", flush=True)
        traceback.print_exc()
    finally:
        files = os.listdir(batchDirectory)
        for file in files:
            os.unlink(os.path.join(batchDirectory, file))
        os.rmdir(batchDirectory)

        del agent

    # This print statement will trigger the parent manager process to kill this process.
    print(datetime.now(), "==== Training Step Completed ====", flush=True)
    return {"trainingStepId": str(trainingStep.id)}


@app.task
def runTrainingStepTask(trainingSequenceId):
    runTrainingStep(trainingSequenceId)



if __name__ == "__main__":
    task = TaskProcess(runTrainingStep)
    task.run()
