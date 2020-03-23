from .celery_config import app
from kwola.components.environments.WebEnvironment import WebEnvironment
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.TestingSequenceModel import TestingSequenceModel
from kwola.models.TrainingStepModel import TrainingStep
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.models.ExecutionTraceModel import ExecutionTrace
from kwola.components.agents.DeepLearningAgent import DeepLearningAgent
from kwola.config.config import getKwolaUserDataDirectory
from kwola.components.TaskProcess import TaskProcess
import concurrent.futures
import random
import numpy
import multiprocessing
import gzip
from datetime import datetime
import traceback
import redis
import json
import scipy.special
import bson
import tempfile
import sklearn.preprocessing
import pickle
import os
from kwola.config import config


def getCacheIdForExecutionTraceId(executionTraceId):
    return "execution-trace-" + str(executionTraceId)


def prepareBatchesForExecutionTrace(executionTraceId, executionSessionId, batchDirectory):
    agentConfiguration = config.getAgentConfiguration()

    sampleCache = redis.Redis(db=3)
    cacheId = getCacheIdForExecutionTraceId(executionTraceId)
    cached = sampleCache.get(cacheId)
    if cached is not None:
        cacheHit = True

        sampleBatch = pickle.loads(gzip.decompress(cached))
    else:
        cacheHit = False

        with open(os.path.join(config.getKwolaUserDataDirectory("models"), "reward_normalizer"), "rb") as normalizerFile:
            trainingRewardNormalizer = pickle.load(normalizerFile)

        executionSession = ExecutionSession.objects(id=bson.ObjectId(executionSessionId)).first()

        agent = DeepLearningAgent(agentConfiguration=agentConfiguration, whichGpu=None)

        batch = agent.prepareBatchForExecutionSession(executionSession, trainingRewardNormalizer=trainingRewardNormalizer)

        sampleBatch = None
        for traceIndex in range(len(executionSession.executionTraces)):
            traceBatch = {}
            for key, value in batch.items():
                traceBatch[key] = value[traceIndex:traceIndex+1]

            cacheId = getCacheIdForExecutionTraceId(str(executionSession.executionTraces[traceIndex].id))

            pickleBytes = pickle.dumps(traceBatch)
            compressedPickleBytes = gzip.compress(pickleBytes)

            # Add small random number here to ensure there isn't a bunch of entries expiring in the cache at the same time
            expiration = agentConfiguration['training_execution_session_cache_expiration_seconds'] + random.randint(0, agentConfiguration['training_execution_session_cache_expiration_max_additional_random_seconds'])
            sampleCache.set(cacheId, compressedPickleBytes, ex=expiration)

            if str(executionSession.executionTraces[traceIndex].id) == executionTraceId:
                sampleBatch = traceBatch

    fileDescriptor, fileName = tempfile.mkstemp(".bin", dir=batchDirectory)

    with open(fileDescriptor, 'wb') as batchFile:
        pickle.dump(sampleBatch, batchFile)

    return fileName, cacheHit


def isNumpyArray(obj):
    return type(obj).__module__ == numpy.__name__


def prepareAndLoadSingleBatchForSubprocess(executionTraces, executionTraceIdMap, batchDirectory, processExecutor, subProcessBatchResultQueue):
    agentConfig = config.getAgentConfiguration()

    traceWeights = numpy.array([trace.lastTrainingRewardLoss for trace in executionTraces])
    countWithLossValue = numpy.count_nonzero(traceWeights)

    if (countWithLossValue / len(executionTraces)) > 0.2:
        traceWeights = numpy.minimum(agentConfig['training_trace_selection_maximum_weight'], traceWeights)
        traceWeights = numpy.maximum(agentConfig['training_trace_selection_minimum_weight'], traceWeights)
    else:
        traceWeights = numpy.ones_like(traceWeights)

    traceProbabilities = scipy.special.softmax(traceWeights)
    traceIds = [trace.id for trace in executionTraces]

    chosenExecutionTraceIds = numpy.random.choice(traceIds, [agentConfig['batch_size']], p=traceProbabilities)

    futures = []
    for traceId in chosenExecutionTraceIds:
        trace = executionTraceIdMap[str(traceId)]

        future = processExecutor.submit(prepareBatchesForExecutionTrace, str(traceId), trace.executionSessionId, batchDirectory)
        futures.append(future)

    cacheHits = []
    samples = []
    for future in futures:
        batchFilename, cacheHit = future.result()
        cacheHits.append(float(cacheHit))

        with open(batchFilename, 'rb') as batchFile:
            sampleBatch = pickle.load(batchFile)
            samples.append(sampleBatch)

        os.unlink(batchFilename)

    batch = {}
    for key in samples[0].keys():
        if isNumpyArray(samples[0][key]):
            batch[key] = numpy.concatenate([sample[key] for sample in samples], axis=0)
        else:
            batch[key] = [sample[key][0] for sample in samples]

    cacheHitRate = numpy.mean(cacheHits)

    resultFileDescriptor, resultFileName = tempfile.mkstemp()
    with open(resultFileDescriptor, 'wb') as file:
        pickle.dump((batch, cacheHitRate), file)

    subProcessBatchResultQueue.put(resultFileName)


def prepareAndLoadBatchesSubprocess(batchDirectory, subProcessCommandQueue, subProcessBatchResultQueue):
    try:
        agentConfig = config.getAgentConfiguration()

        print(datetime.now(), "Starting initialization for batch preparation sub process.", flush=True)

        testSequences = list(TestingSequenceModel.objects(status="completed").no_dereference().order_by('-startTime').only('status', 'startTime', 'executionSessions').limit(agentConfig['training_number_of_recent_testing_sequences_to_use']))
        if len(testSequences) == 0:
            print(datetime.now(), "Error, no test sequences in db to train on for training step.", flush=True)
            return

        # We use this mechanism to force parallel preloading of all the execution traces. Otherwise it just takes forever...
        executionSessionIds = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=agentConfig['training_max_initialization_workers']) as executor:
            executionSessionFutures = []
            for testSequence in testSequences:
                for session in testSequence.executionSessions:
                    executionSessionIds.append(str(session['_ref'].id))
                    executionSessionFutures.append(executor.submit(loadExecutionSession, session['_ref'].id))

            executionSessions = [future.result() for future in executionSessionFutures]

        print(datetime.now(), "Starting loading of execution traces.", flush=True)

        executionTraces = []
        executionTraceIdMap = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=agentConfig['training_max_initialization_workers']) as executor:
            executionTraceFutures = []
            for session in executionSessions:
                for trace in session.executionTraces:
                    executionTraceFutures.append(executor.submit(loadExecutionTrace, trace['_ref'].id))

            for traceFuture in executionTraceFutures:
                trace = pickle.loads(traceFuture.result())
                executionTraces.append(trace)
                executionTraceIdMap[str(trace.id)] = trace

        print(datetime.now(), f"Finished loading of {len(executionTraces)} execution traces.", flush=True)
        print(datetime.now(), "Starting creation of the reward normalizer.", flush=True)

        trainingRewardNormalizer = DeepLearningAgent.createTrainingRewardNormalizer(random.sample(executionSessionIds, min(len(executionSessionIds), agentConfig['training_reward_normalizer_fit_population_size'])))

        with open(os.path.join(config.getKwolaUserDataDirectory("models"), "reward_normalizer"), "wb") as normalizerFile:
            pickle.dump(trainingRewardNormalizer, normalizerFile)

        del testSequences, executionSessionIds, executionSessionFutures, executionSessions, executionTraceFutures

        print(datetime.now(), "Finished creation of the reward normalizer.", flush=True)
        print(datetime.now(), "Finished initialization for batch preparation sub process.", flush=True)

        with concurrent.futures.ProcessPoolExecutor(max_workers=agentConfig['training_max_main_process_workers']) as processExecutor:
            with concurrent.futures.ThreadPoolExecutor(max_workers=agentConfig['training_max_main_thread_workers']) as threadExecutor:
                while True:
                    message, data = subProcessCommandQueue.get()
                    if message == "quit":
                        break
                    elif message == "batch":
                        threadExecutor.submit(prepareAndLoadSingleBatchForSubprocess, executionTraces, executionTraceIdMap, batchDirectory, processExecutor, subProcessBatchResultQueue)
                    elif message == "update-loss":
                        executionTraceId = data["executionTraceId"]
                        sampleRewardLoss = data["sampleRewardLoss"]
                        trace = executionTraceIdMap[executionTraceId]
                        trace.lastTrainingRewardLoss = sampleRewardLoss
                        trace.save()
    except Exception:
        print(datetime.now(), f"Error occurred in the batch preparation sub-process. Exiting.", flush=True)
        traceback.print_exc()


def prepareAndLoadBatch(subProcessCommandQueue, subProcessBatchResultQueue):
    subProcessCommandQueue.put(("batch", {}))

    batchFileName = subProcessBatchResultQueue.get()
    with open(batchFileName, 'rb') as file:
        batch, cacheHit = pickle.load(file)
    os.unlink(batchFileName)

    return batch, cacheHit


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


def loadExecutionSession(sessionId):
    session = ExecutionSession.objects(id=bson.ObjectId(sessionId)).no_dereference().first()
    return session


def loadExecutionTrace(traceId):
    trace = ExecutionTrace.objects(id=bson.ObjectId(traceId)).no_dereference().only('executionSessionId', 'lastTrainingRewardLoss').first()
    return pickle.dumps(trace)


def runTrainingStep(trainingSequenceId):
    try:
        agentConfig = config.getAgentConfiguration()

        print(datetime.now(), "Starting Training Step", flush=True)
        testSequences = list(TestingSequenceModel.objects(status="completed").no_dereference().order_by('-startTime').only('status', 'startTime', 'executionSessions').limit(agentConfig['training_number_of_recent_testing_sequences_to_use']))
        if len(testSequences) == 0:
            print(datetime.now(), "Error, no test sequences in db to train on for training step.", flush=True)
            print(datetime.now(), "==== Training Step Completed ====", flush=True)
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

        subProcessCommandQueue = multiprocessing.Queue()
        subProcessBatchResultQueue = multiprocessing.Queue()
        subProcess = multiprocessing.Process(target=prepareAndLoadBatchesSubprocess, args=(batchDirectory, subProcessCommandQueue, subProcessBatchResultQueue))
        subProcess.start()

    except Exception as e:
        print(datetime.now(), f"Error occurred during initiation of training!", flush=True)
        traceback.print_exc()
        return {}

    try:
        totalBatchesNeeded = agentConfig['iterations_per_training_step'] + int(agentConfig['training_surplus_batches'])
        batchesPrepared = 0

        batchFutures = []

        recentCacheHits = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=agentConfig['training_max_main_thread_workers']) as threadExecutor:
            # First we chuck some batch requests into the queue.
            for n in range(agentConfig['training_precompute_batches_count']):
                batchFutures.append(threadExecutor.submit(prepareAndLoadBatch, subProcessCommandQueue, subProcessBatchResultQueue))
                batchesPrepared += 1

            while trainingStep.numberOfIterationsCompleted < agentConfig['iterations_per_training_step']:
                batch, cacheHitRate = batchFutures.pop(0).result()
                recentCacheHits.append(float(cacheHitRate))

                if batchesPrepared <= totalBatchesNeeded:
                    # Request another session be prepared
                    batchFutures.append(threadExecutor.submit(prepareAndLoadBatch, subProcessCommandQueue, subProcessBatchResultQueue))
                    batchesPrepared += 1

                totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, tracePredictionLoss, \
                    executionFeaturesLoss, targetHomogenizationLoss, predictedCursorLoss, \
                    totalLoss, totalRebalancedLoss, batchReward, \
                    sampleRewardLosses = agent.learnFromBatch(batch)

                for executionTraceId, sampleRewardLoss in zip(batch['traceIds'], sampleRewardLosses):
                    subProcessCommandQueue.put(("update-loss", {"executionTraceId": executionTraceId, "sampleRewardLoss": sampleRewardLoss}))

                trainingStep.presentRewardLosses.append(presentRewardLoss)
                trainingStep.discountedFutureRewardLosses.append(discountedFutureRewardLoss)
                trainingStep.tracePredictionLosses.append(tracePredictionLoss)
                trainingStep.executionFeaturesLosses.append(executionFeaturesLoss)
                trainingStep.targetHomogenizationLosses.append(targetHomogenizationLoss)
                trainingStep.predictedCursorLosses.append(predictedCursorLoss)
                trainingStep.totalRewardLosses.append(totalRewardLoss)
                trainingStep.totalRebalancedLosses.append(totalRebalancedLoss)
                trainingStep.totalLosses.append(totalLoss)
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

        subProcessCommandQueue.put(("quit", {}))

        # Safe guard, don't save the model if any nan's were detected
        if numpy.count_nonzero(numpy.isnan(trainingStep.totalLosses)) == 0:
            agent.save()
            print(datetime.now(), "Agent saved!", flush=True)
        else:
            print(datetime.now(), "ERROR! A NaN was detected in this models output. Not saving model.", flush=True)

    except Exception:
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
