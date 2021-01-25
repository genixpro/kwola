#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


from ...config.logger import getLogger, setupLocalLogging
from ...components.agents.DeepLearningAgent import DeepLearningAgent
from ...components.environments.WebEnvironment import WebEnvironment
from ...tasks.TaskProcess import TaskProcess
from ...config.config import KwolaCoreConfiguration
from ...datamodels.ExecutionSessionModel import ExecutionSession
from ...datamodels.ExecutionTraceModel import ExecutionTrace
from ...datamodels.ExecutionSessionTraceWeights import ExecutionSessionTraceWeights
from ...datamodels.TestingStepModel import TestingStep
from ...datamodels.TrainingStepModel import TrainingStep
from datetime import datetime
import atexit
import subprocess
import concurrent.futures
import gzip
import json
import billiard as multiprocessing
import billiard.pool as multiprocessingpool
import numpy
import os
import pickle
import random
import shutil
import scipy.special
import sys
import tempfile
import time
import torch
import torch.distributed
import traceback
from pprint import pformat
import google.api_core.exceptions
from google.cloud import storage


def isNumpyArray(obj):
    return type(obj).__module__ == numpy.__name__


class TrainingManager:
    def __init__(self, config, trainingSequenceId, trainingStepIndex, gpu=None, coordinatorTempFileName="kwola_distributed_coordinator", testingRunId=None, applicationId=None, gpuWorldSize=torch.cuda.device_count(), plugins=None):
        self.config = KwolaCoreConfiguration(config)

        # Make sure applicationId is set in the config and save it
        if 'applicationId' not in self.config:
            self.config['applicationId'] = applicationId
            self.config.saveConfig()

        self.gpu = gpu
        self.coordinatorTempFileName = coordinatorTempFileName
        self.gpuWorldSize = gpuWorldSize
        self.trainingSequenceId = trainingSequenceId
        self.testingRunId = testingRunId
        self.applicationId = applicationId
        self.trainingStepIndex = trainingStepIndex

        self.trainingStep = None

        self.totalBatchesNeeded = self.config['iterations_per_training_step'] * self.config['batches_per_iteration'] + int(self.config['training_surplus_batches'])
        self.batchesPrepared = 0
        self.batchFutures = []
        self.recentCacheHits = []
        self.starved = False
        self.lastStarveStateAdjustment = 0
        self.coreLearningTimes = []
        self.loopTimes = []

        self.testingSteps = []
        self.agent = None

        self.batchDirectory = None
        self.subProcessCommandQueues = []
        self.subProcessBatchResultQueues = []
        self.subProcesses = []

        if plugins is None:
            self.plugins = []
        else:
            self.plugins = plugins


    def createTrainingStep(self):
        trainingStep = TrainingStep(id=str(self.trainingSequenceId) + "_training_step_" + str(self.trainingStepIndex))
        trainingStep.startTime = datetime.now()
        trainingStep.trainingSequenceId = self.trainingSequenceId
        trainingStep.testingRunId = self.testingRunId
        trainingStep.applicationId = self.applicationId
        trainingStep.status = "running"
        trainingStep.numberOfIterationsCompleted = 0
        trainingStep.presentRewardLosses = []
        trainingStep.discountedFutureRewardLosses = []
        trainingStep.tracePredictionLosses = []
        trainingStep.executionFeaturesLosses = []
        trainingStep.predictedCursorLosses = []
        trainingStep.totalRewardLosses = []
        trainingStep.totalLosses = []
        trainingStep.totalRebalancedLosses = []
        trainingStep.hadNaN = False
        trainingStep.saveToDisk(self.config)
        self.trainingStep = trainingStep

    def initializeGPU(self):
        if self.gpu is not None:
            for subprocessIndex in range(10):
                try:
                    init_method = f"file://{os.path.join(tempfile.gettempdir(), self.coordinatorTempFileName)}"

                    if sys.platform == "win32" or sys.platform == "win64":
                        init_method = f"file:///{os.path.join(tempfile.gettempdir(), self.coordinatorTempFileName)}"

                    torch.distributed.init_process_group(backend="gloo",
                                                         world_size=self.gpuWorldSize,
                                                         rank=self.gpu,
                                                         init_method=init_method)
                    break
                except RuntimeError:
                    time.sleep(1)
                    if subprocessIndex == 9:
                        raise
            torch.cuda.set_device(self.gpu)
            getLogger().info(f"Cuda Ready on GPU {self.gpu}")

    def loadTestingSteps(self):
        self.testingSteps = [step for step in TrainingManager.loadAllTestingSteps(self.config, self.applicationId) if step.status == "completed"]


    def runTraining(self):
        success = True
        exception = None

        try:
            try:
                multiprocessing.set_start_method('spawn')
            except RuntimeError:
                pass
            if self.config['print_configuration_on_startup']:
                getLogger().info(f"Starting Training Step with configuration:\n{pformat(self.config.configData)}")
            else:
                getLogger().info(f"Starting Training Step")

            self.initializeGPU()
            self.createTrainingStep()
            self.loadTestingSteps()

            if len(self.testingSteps) == 0:
                errorMessage = f"Error, no test sequences to train on for training step."
                getLogger().warning(f"{errorMessage}")
                getLogger().info(f"==== Training Step Completed ====")
                return {"success": False, "exception": errorMessage}

            self.agent = DeepLearningAgent(config=self.config, whichGpu=self.gpu)
            self.agent.initialize()
            try:
                self.agent.load()
            except RuntimeError as e:
                getLogger().error(
                    f"Warning! DeepLearningAgent was unable to load the model file from disk, and so is instead using a freshly random initialized neural network. The original error is: {traceback.format_exc()}")
                self.agent.save()

            self.createSubproccesses()

            for plugin in self.plugins:
                plugin.trainingStepStarted(self.trainingStep)

        except Exception as e:
            errorMessage = f"Error occurred during initiation of training! {traceback.format_exc()}"
            getLogger().warning(f"{errorMessage}")
            return {"success": False, "exception": errorMessage}

        try:
            self.threadExecutor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config['training_max_batch_prep_thread_workers'] * self.config['training_batch_prep_subprocesses'])

            self.queueBatchesForPrecomputation()

            while self.trainingStep.numberOfIterationsCompleted < self.config['iterations_per_training_step']:
                loopStart = datetime.now()

                self.updateBatchPrepStarvedState()
                batches = self.fetchBatchesForIteration()

                success = self.learnFromBatches(batches)
                if not success:
                    break

                if self.trainingStep.numberOfIterationsCompleted % self.config['training_update_target_network_every'] == (self.config['training_update_target_network_every'] - 1):
                    getLogger().info(f"Updating the target network weights to the current primary network weights.")
                    self.agent.updateTargetNetwork()

                self.trainingStep.numberOfIterationsCompleted += 1

                if self.trainingStep.numberOfIterationsCompleted % self.config['print_loss_iterations'] == (self.config['print_loss_iterations'] - 1):
                    if self.gpu is None or self.gpu == 0:
                        getLogger().info(f"Completed {self.trainingStep.numberOfIterationsCompleted + 1} batches. Overall average time per batch: {numpy.average(self.loopTimes[-25:]):.3f}. Core learning time: {numpy.average(self.coreLearningTimes[-25:]):.3f}")
                        self.printMovingAverageLosses()
                        if self.config['print_cache_hit_rate']:
                            getLogger().info(f"Batch cache hit rate {100 * numpy.mean(self.recentCacheHits[-self.config['print_cache_hit_rate_moving_average_length']:]):.0f}%")

                if self.trainingStep.numberOfIterationsCompleted % self.config['iterations_between_db_saves'] == (self.config['iterations_between_db_saves'] - 1):
                    if self.gpu is None or self.gpu == 0:
                        self.trainingStep.saveToDisk(self.config)

                for plugin in self.plugins:
                    plugin.iterationCompleted(self.trainingStep)

                loopEnd = datetime.now()
                self.loopTimes.append((loopEnd - loopStart).total_seconds())

            getLogger().info(f"Finished the core training loop. Saving the training step {self.trainingStep.id}")
            self.trainingStep.endTime = datetime.now()
            self.trainingStep.averageTimePerIteration = (self.trainingStep.endTime - self.trainingStep.startTime).total_seconds() / self.trainingStep.numberOfIterationsCompleted
            self.trainingStep.averageLoss = float(numpy.mean(self.trainingStep.totalLosses))
            self.trainingStep.status = "completed"

            for plugin in self.plugins:
                plugin.trainingStepFinished(self.trainingStep)

            self.trainingStep.saveToDisk(self.config)

            self.threadExecutor.shutdown(wait=True)

            self.saveAgent()
            self.shutdownAndJoinSubProcesses()

        except Exception:
            getLogger().error(f"Error occurred while learning sequence!\n{traceback.format_exc()}")
            success = False
            exception = traceback.format_exc()
        finally:
            files = os.listdir(self.batchDirectory)
            for file in files:
                os.unlink(os.path.join(self.batchDirectory, file))
            shutil.rmtree(self.batchDirectory)

            del self.agent

        # This print statement will trigger the parent manager process to kill this process.
        getLogger().info(f"==== Training Step Completed ====")
        returnData = {"trainingStepId": str(self.trainingStep.id), "success": success}
        if exception is not None:
            returnData['exception'] = exception

        return returnData

    def learnFromBatches(self, batches):
        learningIterationStartTime = datetime.now()
        results = self.agent.learnFromBatches(batches)
        learningIterationFinishTime = datetime.now()
        self.coreLearningTimes.append((learningIterationFinishTime - learningIterationStartTime).total_seconds())

        if results is not None:
            for result, batch in zip(results, batches):
                totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, \
                stateValueLoss, advantageLoss, actionProbabilityLoss, tracePredictionLoss, \
                executionFeaturesLoss, predictedCursorLoss, \
                totalLoss, totalRebalancedLoss, batchReward, \
                sampleRewardLosses = result

                self.trainingStep.presentRewardLosses.append(presentRewardLoss)
                self.trainingStep.discountedFutureRewardLosses.append(discountedFutureRewardLoss)
                self.trainingStep.stateValueLosses.append(stateValueLoss)
                self.trainingStep.advantageLosses.append(advantageLoss)
                self.trainingStep.actionProbabilityLosses.append(actionProbabilityLoss)
                self.trainingStep.tracePredictionLosses.append(tracePredictionLoss)
                self.trainingStep.executionFeaturesLosses.append(executionFeaturesLoss)
                self.trainingStep.predictedCursorLosses.append(predictedCursorLoss)
                self.trainingStep.totalRewardLosses.append(totalRewardLoss)
                self.trainingStep.totalRebalancedLosses.append(totalRebalancedLoss)
                self.trainingStep.totalLosses.append(totalLoss)

                for executionTraceId, sampleRewardLoss in zip(batch['traceIds'], sampleRewardLosses):
                    for subProcessCommandQueue in self.subProcessCommandQueues:
                        subProcessCommandQueue.put(
                            ("update-loss", {"executionTraceId": executionTraceId, "sampleRewardLoss": sampleRewardLoss}))
            return True
        else:
            self.trainingStep.hadNaN = True
            return False

    def queueBatchesForPrecomputation(self):
        # First we chuck some batch requests into the queue.
        for n in range(self.config['training_precompute_batches_count']):
            subProcessIndex = (self.batchesPrepared % self.config['training_batch_prep_subprocesses'])
            self.batchFutures.append(
                self.threadExecutor.submit(TrainingManager.prepareAndLoadBatch,
                                           self.subProcessCommandQueues[subProcessIndex],
                                           self.subProcessBatchResultQueues[subProcessIndex]))
            self.batchesPrepared += 1

    def createSubproccesses(self):
        # Haven't decided yet whether we should force Kwola to always write to disc or spool in memory
        # using /tmp. The following lines switch between the two approaches
        # self.batchDirectory = tempfile.mkdtemp(dir=getKwolaUserDataDirectory("batches"))
        self.batchDirectory = tempfile.mkdtemp()

        self.subProcessCommandQueues = []
        self.subProcessBatchResultQueues = []
        self.subProcesses = []

        for subprocessIndex in range(self.config['training_batch_prep_subprocesses']):
            subProcessCommandQueue = multiprocessing.Queue()
            subProcessBatchResultQueue = multiprocessing.Queue()

            subProcess = multiprocessing.Process(target=TrainingManager.prepareAndLoadBatchesSubprocess, args=(self.config.serialize(), self.batchDirectory, subProcessCommandQueue, subProcessBatchResultQueue, subprocessIndex, self.applicationId))
            subProcess.start()
            atexit.register(lambda: subProcess.terminate())

            self.subProcessCommandQueues.append(subProcessCommandQueue)
            self.subProcessBatchResultQueues.append(subProcessBatchResultQueue)
            self.subProcesses.append(subProcess)

        for queue in self.subProcessBatchResultQueues:
            readyState = queue.get()

            if readyState == "error":
                raise Exception("Error occurred during batch prep sub process initiation.")

    def countReadyBatches(self):
        ready = 0
        for future in self.batchFutures:
            if future.done():
                ready += 1
        return ready


    def updateBatchPrepStarvedState(self):
        if self.trainingStep.numberOfIterationsCompleted > (self.lastStarveStateAdjustment + self.config['training_min_batches_between_starve_state_adjustments']):
            ready = self.countReadyBatches()
            if ready < (self.config['training_precompute_batches_count'] / 4):
                if not self.starved:
                    for subProcessCommandQueue in self.subProcessCommandQueues:
                        subProcessCommandQueue.put(("starved", {}))
                    self.starved = True
                    getLogger().info(
                        f"GPU pipeline is starved for batches. Ready batches: {ready}. Switching to starved state.")
                    self.lastStarveStateAdjustment = self.trainingStep.numberOfIterationsCompleted
            else:
                if self.starved:
                    for subProcessCommandQueue in self.subProcessCommandQueues:
                        subProcessCommandQueue.put(("full", {}))
                    self.starved = False
                    getLogger().info(f"GPU pipeline is full of batches. Ready batches: {ready}. Switching to full state")
                    self.lastStarveStateAdjustment = self.trainingStep.numberOfIterationsCompleted

    def fetchBatchesForIteration(self):
        batches = []

        for batchIndex in range(self.config['batches_per_iteration']):
            chosenBatchIndex = 0
            found = False
            for futureIndex, future in enumerate(self.batchFutures):
                if future.done():
                    chosenBatchIndex = futureIndex
                    found = True
                    break

            batchFetchStartTime = datetime.now()
            batch, cacheHitRate = self.batchFutures.pop(chosenBatchIndex).result(timeout=self.config['training_batch_fetch_timeout'])
            batchFetchFinishTime = datetime.now()

            fetchTime = (batchFetchFinishTime - batchFetchStartTime).total_seconds()

            if not found and fetchTime > 0.5:
                getLogger().info(
                    f"I was starved waiting for a batch to be assembled. Waited: {fetchTime:.2f}")

            self.recentCacheHits.append(float(cacheHitRate))
            batches.append(batch)

            if self.batchesPrepared <= self.totalBatchesNeeded:
                # Request another session be prepared
                subProcessIndex = (self.batchesPrepared % self.config['training_batch_prep_subprocesses'])
                self.batchFutures.append(self.threadExecutor.submit(TrainingManager.prepareAndLoadBatch,
                                                                    self.subProcessCommandQueues[subProcessIndex],
                                                                    self.subProcessBatchResultQueues[subProcessIndex]))
                self.batchesPrepared += 1

        return batches


    def shutdownAndJoinSubProcesses(self):
        getLogger().info(f"Shutting down and joining the sub-processes")
        for subProcess, subProcessCommandQueue in zip(self.subProcesses, self.subProcessCommandQueues):
            subProcessCommandQueue.put(("quit", {}))
            getLogger().info(f"Waiting for batch prep subprocess with pid {subProcess.pid} to quit")
            subProcess.join(timeout=30)
            if subProcess.is_alive():
                # Use kill in python 3.7+, terminate in lower versions
                if hasattr(subProcess, 'kill'):
                    getLogger().info(f"Sending the subprocess with pid {subProcess.pid} the kill signal with.")
                    subProcess.kill()
                else:
                    getLogger().info(f"Sending the subprocess with pid {subProcess.pid} the terminate signal.")
                    subProcess.terminate()

    def saveAgent(self):
        # Safe guard, don't save the model if any nan's were detected
        if not self.trainingStep.hadNaN:
            if self.gpu is None or self.gpu == 0:
                getLogger().info(f"Saving the core training model.")
                self.agent.save()
                if self.config['training_save_model_checkpoints']:
                    self.agent.save(saveName=str(self.trainingStep.id))
                getLogger().info(f"Agent saved!")
        else:
            getLogger().error(f"ERROR! A NaN was detected in this models output. Not saving model.")

    @staticmethod
    def saveExecutionSessionTraceWeights(traceWeightData, config):
        config = KwolaCoreConfiguration(config)
        traceWeightData.saveToDisk(config)


    @staticmethod
    def writeSingleExecutionTracePreparedSampleData(traceBatch, config):
        traceId = traceBatch['traceIds'][0]

        cacheFile = traceId + "-sample.pickle.gz"

        pickleBytes = pickle.dumps(traceBatch, protocol=pickle.HIGHEST_PROTOCOL)
        compressedPickleBytes = gzip.compress(pickleBytes)

        config.saveKwolaFileData("prepared_samples", cacheFile, compressedPickleBytes)

    @staticmethod
    def addExecutionSessionToSampleCache(executionSessionId, config):
        getLogger().info(f"Adding {executionSessionId} to the sample cache.")
        config.connectToMongoIfNeeded()
        maxAttempts = 10
        for attempt in range(maxAttempts):
            try:
                agent = DeepLearningAgent(config, whichGpu=None)

                agent.loadSymbolMap()

                executionSession = ExecutionSession.loadFromDisk(executionSessionId, config)

                batches = agent.prepareBatchesForExecutionSession(executionSession)

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = []
                    for traceIndex, traceBatch in zip(range(len(executionSession.executionTraces) - 1), batches):
                        futures.append(executor.submit(TrainingManager.writeSingleExecutionTracePreparedSampleData, traceBatch, config))
                    for future in futures:
                        future.result(timeout=config['training_write_prepared_sample_timeout'])
                getLogger().info(f"Finished adding {executionSessionId} to the sample cache.")
                break
            except Exception as e:
                if attempt == (maxAttempts - 1):
                    getLogger().error(f"Error! Failed to prepare samples for execution session {executionSessionId}. Error was: {traceback.print_exc()}")
                    raise
                else:
                    getLogger().warning(f"Warning! Failed to prepare samples for execution session {executionSessionId}. Error was: {traceback.print_exc()}")

    @staticmethod
    def destroyPreparedSamplesForExecutionTrace(config, executionTraceId):
        """ This method is used to destroy the cached prepared_samples data so that it can later be recreated. This is commonly triggered
            in the event of an error."""
        config = KwolaCoreConfiguration(config)

        cacheFile = executionTraceId + "-sample.pickle.gz"

        # Just for compatibility with the old naming scheme
        oldCacheFileName = executionTraceId + ".pickle.gz"

        config.deleteKwolaFileData("prepared_samples", cacheFile)
        config.deleteKwolaFileData("prepared_samples", oldCacheFileName)

    @staticmethod
    def prepareBatchesForExecutionTrace(config, executionTraceId, executionSessionId, batchDirectory, applicationId):
        try:
            config = KwolaCoreConfiguration(config)

            agent = DeepLearningAgent(config, whichGpu=None)

            cacheFile = executionTraceId + "-sample.pickle.gz"

            fileData = config.loadKwolaFileData("prepared_samples", cacheFile)
            if fileData is None:
                TrainingManager.addExecutionSessionToSampleCache(executionSessionId, config)
                cacheHit = False

                fileData = config.loadKwolaFileData("prepared_samples", cacheFile)
                sampleBatch = pickle.loads(gzip.decompress(fileData))
            else:
                sampleBatch = pickle.loads(gzip.decompress(fileData))
                cacheHit = True

            imageWidth = sampleBatch['processedImages'].shape[3]
            imageHeight = sampleBatch['processedImages'].shape[2]

            # Calculate the crop positions for the main training image
            if config['training_enable_image_cropping']:
                randomXDisplacement = random.randint(-config['training_crop_center_random_x_displacement'], config['training_crop_center_random_x_displacement'])
                randomYDisplacement = random.randint(-config['training_crop_center_random_y_displacement'], config['training_crop_center_random_y_displacement'])

                cropLeft, cropTop, cropRight, cropBottom = agent.calculateTrainingCropPosition(sampleBatch['actionXs'][0] + randomXDisplacement, sampleBatch['actionYs'][0] + randomYDisplacement, imageWidth, imageHeight)
            else:
                cropLeft = 0
                cropRight = imageWidth
                cropTop = 0
                cropBottom = imageHeight

            # Calculate the crop positions for the next state image
            if config['training_enable_next_state_image_cropping']:
                nextStateCropCenterX = random.randint(10, imageWidth - 10)
                nextStateCropCenterY = random.randint(10, imageHeight - 10)

                nextStateCropLeft, nextStateCropTop, nextStateCropRight, nextStateCropBottom = agent.calculateTrainingCropPosition(nextStateCropCenterX, nextStateCropCenterY, imageWidth, imageHeight, nextStepCrop=True)
            else:
                nextStateCropLeft = 0
                nextStateCropRight = imageWidth
                nextStateCropTop = 0
                nextStateCropBottom = imageHeight

            # Crop all the input images and update the action x & action y
            # This is done at this step because the cropping is random
            # and thus you don't want to store the randomly cropped version
            # in the redis cache
            sampleBatch['processedImages'] = sampleBatch['processedImages'][:, :, cropTop:cropBottom, cropLeft:cropRight]
            sampleBatch['pixelActionMaps'] = sampleBatch['pixelActionMaps'][:, :, cropTop:cropBottom, cropLeft:cropRight]
            sampleBatch['rewardPixelMasks'] = sampleBatch['rewardPixelMasks'][:, cropTop:cropBottom, cropLeft:cropRight]
            sampleBatch['actionXs'] = sampleBatch['actionXs'] - cropLeft
            sampleBatch['actionYs'] = sampleBatch['actionYs'] - cropTop

            sampleBatch['nextProcessedImages'] = sampleBatch['nextProcessedImages'][:, :, nextStateCropTop:nextStateCropBottom, nextStateCropLeft:nextStateCropRight]
            sampleBatch['nextPixelActionMaps'] = sampleBatch['nextPixelActionMaps'][:, :, nextStateCropTop:nextStateCropBottom, nextStateCropLeft:nextStateCropRight]

            # Add augmentation to the processed images. This is done at this stage
            # so that we don't store the augmented version in the cache.
            # Instead, we want the pure version in the cache and create a
            # new augmentation every time we load it.
            processedImage = sampleBatch['processedImages'][0]
            augmentedImage = agent.augmentProcessedImageForTraining(processedImage)
            sampleBatch['processedImages'][0] = augmentedImage

            fileDescriptor, fileName = tempfile.mkstemp(".bin", dir=batchDirectory)

            with open(fileDescriptor, 'wb') as batchFile:
                pickle.dump(sampleBatch, batchFile, protocol=pickle.HIGHEST_PROTOCOL)

            return fileName, cacheHit
        except Exception:
            getLogger().critical(traceback.format_exc())
            raise

    @staticmethod
    def prepareAndLoadSingleBatchForSubprocess(config, executionSessionTraceWeightDatas, executionSessionTraceWeightDataIdMap, batchDirectory, cacheFullState, processPool, subProcessCommandQueue, subProcessBatchResultQueue, applicationId):
        try:
            tracesWithWeightObjects = [(traceWeightData.weights[traceId], traceId, traceWeightData)
                                       for traceWeightData in executionSessionTraceWeightDatas
                                       for traceId in traceWeightData.weights]

            traceWeights = numpy.array([weight[0] for weight in tracesWithWeightObjects], dtype=numpy.float64)

            traceWeights = numpy.minimum(config['training_trace_selection_maximum_weight'], traceWeights)
            traceWeights = numpy.maximum(config['training_trace_selection_minimum_weight'], traceWeights)

            if not cacheFullState:
                # We bias the random selection of the algorithm towards whatever
                # is at the one end of the list when we aren't in cache full state.
                # This just gives a bit of bias towards the algorithm to select
                # the same execution traces while the system is booting up
                # and gets the GPU to full speed sooner without requiring the cache to be
                # completely filled. This is helpful when cold starting a training run, such
                # as when doing R&D. it basically plays no role once you have a run going
                # for any length of time since the batch cache will fill up within
                # a single training step.
                traceWeights = traceWeights + numpy.arange(0, config['training_trace_selection_cache_not_full_state_one_side_bias'], len(traceWeights))

            traceProbabilities = numpy.array(traceWeights) / numpy.sum(traceWeights)
            traceIds = [weight[1] for weight in tracesWithWeightObjects]

            chosenExecutionTraceIds = numpy.random.choice(traceIds, [config['batch_size']], p=traceProbabilities)

        except Exception:
            getLogger().error(f"prepareAndLoadSingleBatchForSubprocess failed! Putting a retry into the queue.\n{traceback.format_exc()}")
            subProcessCommandQueue.put(("batch", {}))
            return 1.0

        try:
            futures = []
            for traceId in chosenExecutionTraceIds:
                traceWeightData = executionSessionTraceWeightDataIdMap[str(traceId)]

                future = processPool.apply_async(TrainingManager.prepareBatchesForExecutionTrace, (config.serialize(), str(traceId), str(traceWeightData['id']), batchDirectory, applicationId))
                futures.append(future)

            cacheHits = []
            samples = []
            for future in futures:
                batchFilename, cacheHit = future.get(timeout=config['training_prepare_batches_for_execution_trace_timeout'])
                cacheHits.append(float(cacheHit))

                with open(batchFilename, 'rb') as batchFile:
                    sampleBatch = pickle.load(batchFile)
                    samples.append(sampleBatch)

                os.unlink(batchFilename)

            batch = {}
            for key in samples[0].keys():
                # We have to do something special here since they are not concatenated the normal way
                if key == "symbolIndexes" or key == 'symbolWeights' \
                        or key == "nextSymbolIndexes" or key == 'nextSymbolWeights' \
                        or key == "decayingFutureSymbolIndexes" or key == 'decayingFutureSymbolWeights':
                    batch[key] = numpy.concatenate([sample[key][0] for sample in samples], axis=0)

                    currentOffset = 0
                    offsets = []
                    for sample in samples:
                        offsets.append(currentOffset)
                        currentOffset += len(sample[key][0])

                    if 'next' in key:
                        batch['nextSymbolOffsets'] = numpy.array(offsets)
                    elif 'decaying' in key:
                        batch['decayingFutureSymbolOffsets'] = numpy.array(offsets)
                    else:
                        batch['symbolOffsets'] = numpy.array(offsets)
                else:
                    if isNumpyArray(samples[0][key]):
                        batch[key] = numpy.concatenate([sample[key] for sample in samples], axis=0)
                    else:
                        batch[key] = [sample[key][0] for sample in samples]

            cacheHitRate = numpy.mean(cacheHits)

            resultFileDescriptor, resultFileName = tempfile.mkstemp()
            with open(resultFileDescriptor, 'wb') as file:
                pickle.dump((batch, cacheHitRate), file, protocol=pickle.HIGHEST_PROTOCOL)

            subProcessBatchResultQueue.put(resultFileName)

            return cacheHitRate
        except Exception as e:
            # Both KeyboardInterrupt and FileNotFoundError can occur when you Ctrl-C a process from the terminal.
            # We don't want to force recreating the sample cache just because of that.
            if not isinstance(e, KeyboardInterrupt) and not isinstance(e, FileNotFoundError):
                getLogger().error(f"prepareAndLoadSingleBatchForSubprocess failed! Error: {type(e)}. Destroying the batch cache for the traces and then putting a retry into the queue.\n{traceback.format_exc()}")

                # As a precautionary measure, we destroy whatever data is in the
                # prepared samples cache for all of the various traceIds that were
                # chosen here.
                for traceId in chosenExecutionTraceIds:
                    TrainingManager.destroyPreparedSamplesForExecutionTrace(config.serialize(), traceId)

                subProcessCommandQueue.put(("batch", {}))
            return 1.0

    @staticmethod
    def loadAllTestingSteps(config, applicationId=None):
        if config['data_serialization_method']['default'] == 'mongo':
            return list(TestingStep.objects(applicationId=applicationId).no_dereference())
        else:
            testStepsDir = config.getKwolaUserDataDirectory("testing_steps")

            testingSteps = []

            for fileName in os.listdir(testStepsDir):
                if ".lock" not in fileName:
                    stepId = fileName
                    stepId = stepId.replace(".json", "")
                    stepId = stepId.replace(".gz", "")
                    stepId = stepId.replace(".pickle", "")

                    testingSteps.append(TestingStep.loadFromDisk(stepId, config))

            return testingSteps

    @staticmethod
    def updateTraceRewardLoss(traceId, applicationId, sampleRewardLoss, config):
        config = KwolaCoreConfiguration(config)
        trace = ExecutionTrace.loadFromDisk(traceId, config, omitLargeFields=False, applicationId=applicationId)
        trace.lastTrainingRewardLoss = sampleRewardLoss
        trace.saveToDisk(config)

    @staticmethod
    def prepareAndLoadBatchesSubprocess(config, batchDirectory, subProcessCommandQueue, subProcessBatchResultQueue, subprocessIndex=0, applicationId=None):
        try:
            setupLocalLogging(config)

            config = KwolaCoreConfiguration(config)

            getLogger().info(f"Starting initialization for batch preparation sub process.")

            testingSteps = sorted([step for step in TrainingManager.loadAllTestingSteps(config, applicationId) if step.status == "completed"], key=lambda step: step.startTime, reverse=True)
            testingSteps = list(testingSteps)[:int(config['training_number_of_recent_testing_sequences_to_use'])]

            if len(testingSteps) == 0:
                getLogger().warning(f"Error, no test sequences to train on for training step.")
                subProcessBatchResultQueue.put("error")
                return
            else:
                getLogger().info(f"Found {len(testingSteps)} total testing steps for this application.")

            # We use this mechanism to force parallel preloading of all the execution traces. Otherwise it just takes forever...
            executionSessionIds = []
            executionSessionCount = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(config['training_max_initialization_workers'] / config['training_batch_prep_subprocesses'])) as executor:
                executionSessionFutures = []
                for testStepIndex, testStep in enumerate(testingSteps):
                    for sessionId in testStep.executionSessions:
                        executionSessionCount += 1
                        if executionSessionCount % config['training_batch_prep_subprocesses'] == subprocessIndex:
                            executionSessionIds.append(str(sessionId))
                            executionSessionFutures.append(executor.submit(TrainingManager.loadExecutionSession, sessionId, config))

                executionSessions = [future.result(timeout=config['training_execution_session_fetch_timeout']) for future in executionSessionFutures]

            getLogger().info(f"Found {len(executionSessionIds)} total execution sessions that can be learned.")

            getLogger().info(f"Starting loading of execution trace weight datas.")

            allWindowSizes = list(set(session.windowSize for session in executionSessions))

            executionSessionTraceWeightDatasBySize = {
                windowSize: []
                for windowSize in allWindowSizes
            }
            executionSessionTraceWeightDataIdMap = {}

            initialDataLoadProcessPool = multiprocessingpool.Pool(processes=int(config['training_max_initialization_workers'] / config['training_batch_prep_subprocesses']), initializer=setupLocalLogging)

            executionSessionTraceWeightFutures = []
            for session in executionSessions:
                future = initialDataLoadProcessPool.apply_async(TrainingManager.loadExecutionSessionTraceWeights, [session.id, session.executionTraces[:-1], config.serialize()])
                executionSessionTraceWeightFutures.append((future, session))

            completed = 0
            totalTraces = 0
            for weightFuture, executionSession in executionSessionTraceWeightFutures:
                traceWeightData = None
                try:
                    traceWeightDataStr = weightFuture.get(timeout=60)
                    if traceWeightDataStr is not None:
                        traceWeightData = pickle.loads(traceWeightDataStr)
                except multiprocessing.context.TimeoutError:
                    pass

                if traceWeightData is not None:
                    executionSessionTraceWeightDatasBySize[executionSession.windowSize].append(traceWeightData)
                    for traceId in traceWeightData.weights:
                        executionSessionTraceWeightDataIdMap[str(traceId)] = traceWeightData
                    totalTraces += len(traceWeightData.weights)
                completed += 1
                if completed % 100 == 0:
                    getLogger().info(f"Finished loading {completed} execution trace weight datas.")

            initialDataLoadProcessPool.close()
            initialDataLoadProcessPool.join()
            del initialDataLoadProcessPool

            getLogger().info(f"Finished loading of weight datas for {len(executionSessions)} execution sessions with {totalTraces} combined traces.")

            del testingSteps, executionSessionIds, executionSessionFutures, executionSessions, executionSessionTraceWeightFutures
            getLogger().info(f"Finished initialization for batch preparation sub process.")

            if len(executionSessionTraceWeightDataIdMap) == 0:
                subProcessBatchResultQueue.put("error")
                raise RuntimeError("There are no execution sessions to process in the algorithm.")

            processPool = multiprocessingpool.Pool(processes=config['training_initial_batch_prep_workers'], initializer=setupLocalLogging)
            backgroundTraceSaveProcessPool = multiprocessingpool.Pool(processes=config['training_background_trace_save_workers'], initializer=setupLocalLogging)
            executionSessionTraceWeightSaveFutures = {}

            batchCount = 0
            cacheFullState = True

            lastProcessPool = None
            lastProcessPoolFutures = []
            currentProcessPoolFutures = []

            needToResetPool = False
            starved = False

            subProcessBatchResultQueue.put("ready")

            cacheRateFutures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=config['training_max_batch_prep_thread_workers']) as threadExecutor:
                while True:
                    message, data = subProcessCommandQueue.get()
                    if message == "quit":
                        break
                    elif message == "starved":
                        starved = True
                        needToResetPool = True
                    elif message == "full":
                        starved = False
                        needToResetPool = True
                    elif message == "batch":
                        # See if we need to refresh the process pool. This is done to make sure resources get let go and to be able to switch between the smaller and larger process pool
                        # Depending on the cache hit rate
                        if batchCount % config['training_reset_workers_every_n_batches'] == (config['training_reset_workers_every_n_batches'] - 1):
                            needToResetPool = True

                        windowSize = random.choice(allWindowSizes)

                        future = threadExecutor.submit(TrainingManager.prepareAndLoadSingleBatchForSubprocess, config, executionSessionTraceWeightDatasBySize[windowSize],
                                                       executionSessionTraceWeightDataIdMap, batchDirectory, cacheFullState, processPool, subProcessCommandQueue,
                                                       subProcessBatchResultQueue, applicationId)
                        cacheRateFutures.append(future)
                        currentProcessPoolFutures.append(future)

                        batchCount += 1
                    elif message == "update-loss":
                        executionTraceId = data["executionTraceId"]
                        sampleRewardLoss = data["sampleRewardLoss"]
                        if executionTraceId in executionSessionTraceWeightDataIdMap:
                            traceWeightData = executionSessionTraceWeightDataIdMap[executionTraceId]
                            traceWeightData.weights[executionTraceId] = sampleRewardLoss

                    if needToResetPool and lastProcessPool is None:
                        needToResetPool = False

                        timeout = config['training_batch_fetch_timeout']
                        cacheRates = [
                            future.result(timeout=timeout)
                            for future in cacheRateFutures[-config['training_cache_full_state_moving_average_length']:]
                            if future.done()
                        ]

                        # If the cache is full and the main process isn't starved for batches, we shrink the process pool down to a smaller size.
                        if numpy.mean(cacheRates) > config['training_cache_full_state_min_cache_hit_rate'] and not starved:
                            lastProcessPool = processPool
                            lastProcessPoolFutures = list(currentProcessPoolFutures)
                            currentProcessPoolFutures = []

                            getLogger().debug(f"Resetting batch prep process pool. Cache full state. New workers: {config['training_cache_full_batch_prep_workers']}")

                            processPool = multiprocessingpool.Pool(processes=config['training_cache_full_batch_prep_workers'], initializer=setupLocalLogging)

                            cacheFullState = True
                        # Otherwise we have a full sized process pool so we can plow through all the results.
                        else:
                            lastProcessPool = processPool
                            lastProcessPoolFutures = list(currentProcessPoolFutures)
                            currentProcessPoolFutures = []

                            getLogger().debug(f"Resetting batch prep process pool. Cache starved state. New workers: {config['training_max_batch_prep_workers']}")

                            processPool = multiprocessingpool.Pool(processes=config['training_max_batch_prep_workers'], initializer=setupLocalLogging)

                            cacheFullState = False

                    if lastProcessPool is not None:
                        all = True
                        for future in lastProcessPoolFutures:
                            if not future.done():
                                all = False
                                break
                        if all:
                            lastProcessPool.terminate()
                            lastProcessPool = None
                            lastProcessPoolFutures = []


            for traceWeightData in executionSessionTraceWeightDataIdMap.values():
                backgroundTraceSaveProcessPool.apply_async(TrainingManager.saveExecutionSessionTraceWeights,
                                                                             (traceWeightData, config.serialize()))

            backgroundTraceSaveProcessPool.close()
            backgroundTraceSaveProcessPool.join()
            processPool.terminate()
            if lastProcessPool is not None:
                lastProcessPool.terminate()

        except Exception:
            getLogger().error(f"Error occurred in the batch preparation sub-process. Exiting. {traceback.format_exc()}")

    @staticmethod
    def prepareAndLoadBatch(subProcessCommandQueue, subProcessBatchResultQueue):
        subProcessCommandQueue.put(("batch", {}))

        batchFileName = subProcessBatchResultQueue.get()
        with open(batchFileName, 'rb') as file:
            batch, cacheHit = pickle.load(file)
        os.unlink(batchFileName)

        return batch, cacheHit

    def printMovingAverageLosses(self):
        movingAverageLength = int(self.config['print_loss_moving_average_length'])

        averageStart = max(0, min(len(self.trainingStep.totalRewardLosses) - 1, movingAverageLength))

        averageTotalRewardLoss = numpy.mean(self.trainingStep.totalRewardLosses[-averageStart:])
        averagePresentRewardLoss = numpy.mean(self.trainingStep.presentRewardLosses[-averageStart:])
        averageDiscountedFutureRewardLoss = numpy.mean(self.trainingStep.discountedFutureRewardLosses[-averageStart:])

        averageStateValueLoss = numpy.mean(self.trainingStep.stateValueLosses[-averageStart:])
        averageAdvantageLoss = numpy.mean(self.trainingStep.advantageLosses[-averageStart:])
        averageActionProbabilityLoss = numpy.mean(self.trainingStep.actionProbabilityLosses[-averageStart:])

        averageTracePredictionLoss = numpy.mean(self.trainingStep.tracePredictionLosses[-averageStart:])
        averageExecutionFeatureLoss = numpy.mean(self.trainingStep.executionFeaturesLosses[-averageStart:])
        averagePredictedCursorLoss = numpy.mean(self.trainingStep.predictedCursorLosses[-averageStart:])
        averageTotalLoss = numpy.mean(self.trainingStep.totalLosses[-averageStart:])
        averageTotalRebalancedLoss = numpy.mean(self.trainingStep.totalRebalancedLosses[-averageStart:])

        message = f"Losses:\n"

        message += f"    Moving Average Total Reward Loss: {averageTotalRewardLoss:.6f}\n"
        message += f"    Moving Average Present Reward Loss: {averagePresentRewardLoss:.6f}\n"
        message += f"    Moving Average Discounted Future Reward Loss: {averageDiscountedFutureRewardLoss:.6f}\n"
        message += f"    Moving Average State Value Loss: {averageStateValueLoss:.6f}\n"
        message += f"    Moving Average Advantage Loss: {averageAdvantageLoss:.6f}\n"
        message += f"    Moving Average Action Probability Loss: {averageActionProbabilityLoss:.6f}\n"
        if self.config['enable_trace_prediction_loss']:
            message += f"    Moving Average Trace Prediction Loss: {averageTracePredictionLoss:.6f}\n"
        if self.config['enable_execution_feature_prediction_loss']:
            message += f"    Moving Average Execution Feature Loss: {averageExecutionFeatureLoss:.6f}\n"
        if self.config['enable_cursor_prediction_loss']:
            message += f"    Moving Average Predicted Cursor Loss: {averagePredictedCursorLoss:.6f}\n"

        message += f"    Moving Average Total Loss: {averageTotalLoss:.6f}"
        getLogger().info(message)

    @staticmethod
    def loadExecutionSession(sessionId, config):
        session = ExecutionSession.loadFromDisk(sessionId, config)
        if session is None:
            getLogger().error(f"Error! Did not find execution session {sessionId}")

        return session

    @staticmethod
    def loadExecutionTrace(traceId, applicationId, config):
        config = KwolaCoreConfiguration(config)
        trace = ExecutionTrace.loadFromDisk(traceId, config, omitLargeFields=True, applicationId=applicationId)
        return pickle.dumps(trace, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loadExecutionSessionTraceWeights(sessionId, traceIds, config):
        try:
            config = KwolaCoreConfiguration(config)

            traceWeights = ExecutionSessionTraceWeights.loadFromDisk(sessionId, config, printErrorOnFailure=False)
            if traceWeights is None:
                traceWeights = ExecutionSessionTraceWeights(
                    id=sessionId,
                    weights={traceId: config['training_trace_selection_maximum_weight'] for traceId in traceIds}
                )
                traceWeights.saveToDisk(config)

            return pickle.dumps(traceWeights, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            getLogger().error(traceback.format_exc())
            return None
