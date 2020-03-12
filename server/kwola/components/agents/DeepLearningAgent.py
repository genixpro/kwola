from .BaseAgent import BaseAgent
import math, random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import scipy.signal
import pandas
import cv2
import os
import pickle
import tempfile
import bz2
import os.path
from kwola.models.actions.ClickTapAction import ClickTapAction
import os.path
import numpy
import skimage
import skimage.transform
import skimage.color
import skimage.segmentation
import concurrent.futures
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.actions.RightClickAction import RightClickAction
from kwola.models.actions.TypeAction import TypeAction
from kwola.models.actions.WaitAction import WaitAction
import itertools
from kwola.config import config

from .BradNet import BradNet

def grouper(n, iterable):
    """Chunks an iterable into sublists"""
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk

class DeepLearningAgent(BaseAgent):
    """
        This class represents a deep learning agent, which uses reinforcement learning to make the automated testing more effective
    """

    def __init__(self, agentConfiguration, whichGpu="all"):
        super().__init__()

        self.agentConfiguration = agentConfiguration
        
        self.whichGpu = whichGpu
        self.variableWrapperFunc = (lambda x:x.cuda()) if whichGpu is not None else (lambda x:x)

        self.modelPath = os.path.join(config.getKwolaUserDataDirectory("models"), "deep_learning_model")

        self.cursors = [
            "alias",
            "all-scroll",
            "auto",
            "cell",
            "context-menu",
            "col-resize",
            "copy",
            "crosshair",
            "default",
            "e-resize",
            "ew-resize",
            "grab",
            "grabbing",
            "help",
            "move",
            "n-resize",
            "ne-resize",
            "nesw-resize",
            "ns-resize",
            "nw-resize",
            "nwse-resize",
            "no-drop",
            "none",
            "not-allowed",
            "pointer",
            "progress",
            "row-resize",
            "s-resize",
            "se-resize",
            "sw-resize",
            "text",
            "url",
            "w-resize",
            "wait",
            "zoom-in",
            "zoom-out",
            "none"
        ]

        self.trainingLosses = {
            "totalRewardLoss": [],
            "presentRewardLoss": [],
            "discountedFutureRewardLoss": [],
            "tracePredictionLoss": [],
            "predictedExecutionFeaturesLoss": [],
            "targetHomogenizationLoss": [],
            "predictedCursorLoss": [],
            "totalLoss": [],
            "totalRebalancedLoss": [],
            "batchReward": []
        }

    def load(self):
        """
            Loads the agent from db / disk

            :return:
        """

        if os.path.exists(self.modelPath):
            if self.whichGpu is None:
                device = torch.device('cpu')
                self.model.load_state_dict(torch.load(self.modelPath, map_location=device))
            elif self.whichGpu != "all":
                device = torch.device(f"cuda:{self.whichGpu}")
                self.model.load_state_dict(torch.load(self.modelPath, map_location=device))
            else:
                self.model.load_state_dict(torch.load(self.modelPath))

    def save(self):
        """
            Saves the agent to the db / disk.

            :return:
        """
        torch.save(self.model.state_dict(), self.modelPath)


    def initialize(self, environment):
        """
        Initialize the agent for operating in the given environment.

        :param environment:
        :return:
        """
        self.environment = environment

        rect = environment.screenshotSize()

        self.model = BradNet(self.agentConfiguration, self.environment.branchFeatureSize() * 2, len(self.actions), self.environment.branchFeatureSize(), 12, len(self.cursors), whichGpu=self.whichGpu)

        if self.whichGpu == "all":
            self.model = self.model.cuda()
        elif self.whichGpu is None:
            self.model = self.model.cpu()
        else:
            self.model = self.model.to(torch.device(f"cuda:{self.whichGpu}"))
        # self.model = self.model

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)


    def getImage(self):
        images = self.environment.getImages()

        convertedImageFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for image in images:
                convertedImageFuture = executor.submit(processRawImageParallel, image)
                convertedImageFutures.append(convertedImageFuture)

        convertedImages = [
            convertedImageFuture.result() for convertedImageFuture in convertedImageFutures
        ]

        return numpy.array(convertedImages)

    def getAdditionalFeatures(self):
        branchFeature = self.environment.getBranchFeatures()
        decayingExecutionTraceFeature = self.environment.getExecutionTraceFeatures()

        return numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=1)

    def nextBestActions(self):
        """
            Return the next best action predicted by the agent.

            # :param environment:
            :return:
        """

        if random.random() > self.agentConfiguration['epsilon']:
            images = self.getImage()

            width = images.shape[3]
            height = images.shape[2]

            images = self.variableWrapperFunc(torch.FloatTensor(images))
            additionalFeatures = self.variableWrapperFunc(torch.FloatTensor(self.getAdditionalFeatures()))

            presentRewardPredictions, discountedFutureRewardPredictions, predictedTrace, predictedExecutionFeatures, predictedCursor, predictedPixelFeatures = self.model({"image": images, "additionalFeature": additionalFeatures})

            totalRewardPredictions = presentRewardPredictions + discountedFutureRewardPredictions

            # cv2.imshow('image', numpy.array(q_values[0, 0, :, :]))
            # cv2.waitKey(50)

            actionIndexes = totalRewardPredictions.reshape([-1, width * height * len(self.actionsSorted)]).argmax(1).data

            actionInfoList = [
                BradNet.actionIndexToActionDetails(width, height, len(self.actionsSorted), actionIndex)
                for actionIndex in actionIndexes
            ]

        else:
            actionInfoList = []

            width = self.environment.screenshotSize()['width']
            height = self.environment.screenshotSize()['height']

            for n in range(self.environment.numberParallelSessions()):
                actionType = random.randrange(0, len(self.actionsSorted))
                actionX = random.randrange(0, width)
                actionY = random.randrange(0, height)

                actionInfo = (actionType, actionX, actionY)
                actionInfoList.append(actionInfo)

        actions = [
            self.actions[self.actionsSorted[actionInfo[0]]](actionInfo[1], actionInfo[2])
            for actionInfo in actionInfoList
        ]

        return actions


    def prepareBatchesForExecutionSession(self, testingSequence, executionSession):
        """
            This function prepares batches that can be fed to the neural network.

            :param testingSequence:
            :param executionSession:
            :return:
        """
        frames = []

        videoPath = config.getKwolaUserDataDirectory("videos")
        cap = cv2.VideoCapture(os.path.join(videoPath, f'{str(testingSequence.id)}-{executionSession.tabNumber}.mp4'))

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                image = skimage.color.rgb2hsv(frame[:, :, :3])

                swapped = numpy.swapaxes(numpy.swapaxes(image, 0, 2), 1, 2)

                hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)

                frames.append(hueLightnessImage)
            else:
                break

        # First compute the present reward at each time step
        presentRewards = []
        for trace in executionSession.executionTraces:
            tracePresentReward = 0.0

            if trace.didActionSucceed:
                tracePresentReward += self.agentConfiguration['reward_action_success']
            else:
                tracePresentReward += self.agentConfiguration['reward_action_failure']

            if trace.didCodeExecute:
                tracePresentReward += self.agentConfiguration['reward_code_executed']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_code_executed']

            if trace.didNewBranchesExecute:
                tracePresentReward += self.agentConfiguration['reward_new_code_executed']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_new_code_executed']

            if trace.hadNetworkTraffic:
                tracePresentReward += self.agentConfiguration['reward_network_traffic']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_network_traffic']

            if trace.hadNewNetworkTraffic:
                tracePresentReward += self.agentConfiguration['reward_new_network_traffic']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_new_network_traffic']

            if trace.didScreenshotChange:
                tracePresentReward += self.agentConfiguration['reward_screenshot_changed']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_screenshot_change']

            if trace.isScreenshotNew:
                tracePresentReward += self.agentConfiguration['reward_new_screenshot']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_new_screenshot']

            if trace.didURLChange:
                tracePresentReward += self.agentConfiguration['reward_url_changed']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_url_change']

            if trace.isURLNew:
                tracePresentReward += self.agentConfiguration['reward_new_url']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_new_url']

            if trace.hadLogOutput:
                tracePresentReward += self.agentConfiguration['reward_log_output']
            else:
                tracePresentReward += self.agentConfiguration['reward_no_log_output']

            presentRewards.append(tracePresentReward)

        # Now compute the discounted reward
        discountedFutureRewards = []
        presentRewards.reverse()
        current = 0
        for reward in presentRewards:
            current *= self.agentConfiguration['reward_discount_rate']
            discountedFutureRewards.append(current)
            current += reward

        discountedFutureRewards.reverse()
        presentRewards.reverse()

        # Create the decaying future execution trace for the prediction algorithm
        tracesReversed = list(executionSession.executionTraces)
        tracesReversed.reverse()
        currentTrace = numpy.zeros_like(executionSession.executionTraces[0].branchExecutionTrace)
        executionTraceDiscountRate = self.agentConfiguration['future_execution_trace_decay_rate']
        executionTraces = []
        for trace in tracesReversed:
            executionTrace = numpy.array(trace.branchExecutionTrace)
            currentTrace *= executionTraceDiscountRate
            currentTrace += numpy.minimum(executionTrace, numpy.ones_like(executionTrace))
            executionTraces.append(executionTrace)

        executionTraces.reverse()

        shuffledTraceFrameList = list(zip(executionSession.executionTraces, frames, discountedFutureRewards, presentRewards, executionTraces))
        random.shuffle(shuffledTraceFrameList)

        batches = []

        for batch in grouper(self.agentConfiguration['batch_size'], shuffledTraceFrameList):
            batchFrames = []
            batchAdditionalFeatures = []
            batchActionTypes = []
            batchActionXs = []
            batchActionYs = []
            batchActionIndexes = []
            batchExecutionTraces = []
            batchDiscountedFutureRewards = []
            batchPresentRewards = []
            batchRewardPixelMasks = []
            batchExecutionFeatures = []
            batchCursors = []

            for trace, frame, discountedFutureReward, presentReward, executionTrace in batch:
                width = frame.shape[2]
                height = frame.shape[1]

                branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
                decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
                additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

                batchFrames.append(frame)
                batchAdditionalFeatures.append(additionalFeature)

                action_index = BradNet.actionDetailsToActionIndex(width, height, len(self.actionsSorted), self.actionsSorted.index(trace.actionPerformed.type), trace.actionPerformed.x, trace.actionPerformed.y)

                batchActionTypes.append(trace.actionPerformed.type)
                batchActionXs.append(trace.actionPerformed.x)
                batchActionYs.append(trace.actionPerformed.y)
                batchActionIndexes.append(action_index)

                cursorVector = [0] * len(self.cursors)
                if trace.cursor in self.cursors:
                    cursorVector[self.cursors.index(trace.cursor)] = 1
                else:
                    cursorVector[self.cursors.index("none")] = 1

                batchCursors.append(cursorVector)

                executionFeatures = [
                    trace.didActionSucceed,
                    trace.didErrorOccur,
                    trace.didNewErrorOccur,
                    trace.didCodeExecute,
                    trace.didNewBranchesExecute,
                    trace.hadNetworkTraffic,
                    trace.hadNewNetworkTraffic,
                    trace.didScreenshotChange,
                    trace.isScreenshotNew,
                    trace.didURLChange,
                    trace.isURLNew,
                    trace.hadLogOutput,
                ]

                batchExecutionTraces.append(executionTrace)

                batchDiscountedFutureRewards.append(discountedFutureReward)
                batchPresentRewards.append(presentReward)

                # We use flood-segmentation on the original image to select which pixels we will update reward values for.
                # This works great on UIs because the elements always have big areas of solid-color which respond in the same
                # way.
                rewardPixelMask = skimage.segmentation.flood(frame[1], (int(trace.actionPerformed.y), int(trace.actionPerformed.x)))
                batchRewardPixelMasks.append(rewardPixelMask)

                batchExecutionFeatures.append(executionFeatures)

            # Append an array with all the data to the list of batches.
            # Add the same time we down-sample some of the data points to be more compact.
            # We don't need a high precision for the image itself
            batches.append({
                "frames": numpy.array(batchFrames, dtype=numpy.float16),
                "additionalFeatures": numpy.array(batchAdditionalFeatures, dtype=numpy.float16),
                "actionTypes": batchActionTypes,
                "actionXs": numpy.array(batchActionXs, dtype=numpy.int16),
                "actionYs": numpy.array(batchActionYs, dtype=numpy.int16),
                "actionIndexes": numpy.array(batchActionIndexes, dtype=numpy.int32),
                "executionTraces": numpy.array(batchExecutionTraces, dtype=numpy.int8),
                "discountedFutureRewards": numpy.array(batchDiscountedFutureRewards, dtype=numpy.float32),
                "presentRewards": numpy.array(batchPresentRewards, dtype=numpy.float32),
                "rewardPixelMasks": numpy.array(batchRewardPixelMasks, dtype=numpy.uint8),
                "executionFeatures": numpy.array(batchExecutionFeatures, dtype=numpy.uint8),
                "cursors": numpy.array(batchCursors, dtype=numpy.uint8)
            })
            # print("Finished preparing batch #", len(batches), "for", str(executionSession.id), flush=True)

        # print("Finished preparing all batches for", str(executionSession.id), flush=True)

        return batches

    def learnFromBatch(self, batch):
        """
            Runs backprop on the neural network with the given batch.

            :param batch: A batch of image/action/output pairs. Should be the return value from prepareBatchesForTestingSequence
            :return:
        """
        presentRewardPredictions, discountedFutureRewardPredictions, predictedTraces, predictedExecutionFeatures, predictedCursors, predictedPixelFeatures = self.model({
            "image": self.variableWrapperFunc(torch.FloatTensor(numpy.array(batch['frames']))),
            "additionalFeature": self.variableWrapperFunc(torch.FloatTensor(batch['additionalFeatures'])),
            "action_type": batch['actionTypes'],
            "action_x": batch['actionXs'],
            "action_y": batch['actionYs']
        })

        totalRewardLosses = []
        presentRewardLosses = []
        targetHomogenizationLosses = []
        discountedFutureRewardLosses = []

        for presentRewardImage, discountedFutureRewardImage, pixelFeatureImage, rewardPixelMask, presentReward, discountedFutureReward, actionType in zip(presentRewardPredictions, discountedFutureRewardPredictions, predictedPixelFeatures, batch['rewardPixelMasks'], batch['presentRewards'], batch['discountedFutureRewards'], batch['actionTypes']):
            # if len(totalRewardLosses) == 0:
            #     cv2.imshow('image', rewardPixelMask * 200)
            #     cv2.waitKey(50)

            rewardPixelMask = self.variableWrapperFunc(torch.IntTensor(rewardPixelMask))
            # actionType = self.variableWrapperFunc(torch.IntTensor(actionType))

            presentRewardsMasked = presentRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask
            discountedFutureRewardsMasked = discountedFutureRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask

            torchBatchPresentRewards = torch.ones_like(presentRewardImage[self.actionsSorted.index(actionType)]) * self.variableWrapperFunc(torch.FloatTensor([presentReward])) * rewardPixelMask
            torchBatchDiscountedFutureRewards = torch.ones_like(presentRewardImage[self.actionsSorted.index(actionType)]) * self.variableWrapperFunc(torch.FloatTensor([discountedFutureReward])) * rewardPixelMask

            countPixelMask = (rewardPixelMask.sum())

            presentRewardLoss = (presentRewardsMasked - torchBatchPresentRewards).pow(2).sum() / countPixelMask
            discountedFutureRewardLoss = (discountedFutureRewardsMasked - torchBatchDiscountedFutureRewards).pow(2).sum() / countPixelMask

            # Target Homogenization loss - basically, all of the features for the masked area should produce similar features
            pixelFeaturesImageMasked = pixelFeatureImage * rewardPixelMask
            averageFeatures = (pixelFeaturesImageMasked.sum(1).sum(1) / countPixelMask).unsqueeze(1).unsqueeze(1)
            targetHomogenizationLoss = ((pixelFeaturesImageMasked - averageFeatures) * rewardPixelMask).pow(2).sum() / (countPixelMask * self.agentConfiguration['pixel_features'])

            sampleLoss = presentRewardLoss + discountedFutureRewardLoss
            totalRewardLosses.append(sampleLoss.unsqueeze(0))

            presentRewardLosses.append(presentRewardLoss.unsqueeze(0))
            discountedFutureRewardLosses.append(discountedFutureRewardLoss.unsqueeze(0))

            targetHomogenizationLosses.append(targetHomogenizationLoss.unsqueeze(0))

        tracePredictionLoss = (predictedTraces - self.variableWrapperFunc(torch.FloatTensor(batch['executionTraces']))).abs().mean()

        predictedExecutionFeaturesLoss = (predictedExecutionFeatures - self.variableWrapperFunc(torch.FloatTensor(batch['executionFeatures']))).abs().mean()

        predictedCursorLoss = (predictedCursors - self.variableWrapperFunc(torch.FloatTensor(batch['cursors']))).abs().mean()

        totalRewardLoss = torch.mean(torch.cat(totalRewardLosses))
        presentRewardLoss = torch.mean(torch.cat(presentRewardLosses))
        discountedFutureRewardLoss = torch.mean(torch.cat(discountedFutureRewardLosses))
        targetHomogenizationLoss = torch.mean(torch.cat(targetHomogenizationLosses))

        totalLoss = totalRewardLoss + tracePredictionLoss + predictedExecutionFeaturesLoss + targetHomogenizationLoss + predictedCursorLoss

        if len(self.trainingLosses['totalLoss']) > 1:
            runningAverageRewardLoss = numpy.mean(self.trainingLosses['totalRewardLoss'][-self.agentConfiguration['loss_balancing_moving_average_period']:])
            runningAverageTracePredictionLoss = numpy.mean(self.trainingLosses['tracePredictionLoss'][-self.agentConfiguration['loss_balancing_moving_average_period']:])
            runningAverageExecutionFeaturesLoss = numpy.mean(self.trainingLosses['predictedExecutionFeaturesLoss'][-self.agentConfiguration['loss_balancing_moving_average_period']:])
            runningAverageHomogenizationLoss = numpy.mean(self.trainingLosses['targetHomogenizationLoss'][-self.agentConfiguration['loss_balancing_moving_average_period']:])
            runningAveragePredictedCursorLoss = numpy.mean(self.trainingLosses['predictedCursorLoss'][-self.agentConfiguration['loss_balancing_moving_average_period']:])

            tracePredictionAdustment = (runningAverageRewardLoss / runningAverageTracePredictionLoss) * self.agentConfiguration['loss_ratio_trace_prediction']
            executionFeaturesAdustment = (runningAverageRewardLoss / runningAverageExecutionFeaturesLoss) * self.agentConfiguration['loss_ratio_execution_features']
            homogenizationAdustment = (runningAverageRewardLoss / runningAverageHomogenizationLoss) * self.agentConfiguration['loss_ratio_homogenization']
            predictedCursorAdustment = (runningAverageRewardLoss / runningAveragePredictedCursorLoss) * self.agentConfiguration['loss_ratio_predicted_cursor']
        else:
            tracePredictionAdustment = 1
            executionFeaturesAdustment = 1
            homogenizationAdustment = 1
            predictedCursorAdustment = 1

        totalRebalancedLoss = totalRewardLoss + \
                              tracePredictionLoss * self.variableWrapperFunc(torch.FloatTensor([tracePredictionAdustment])) + \
                              predictedExecutionFeaturesLoss * self.variableWrapperFunc(torch.FloatTensor([executionFeaturesAdustment])) + \
                              targetHomogenizationLoss * self.variableWrapperFunc(torch.FloatTensor([homogenizationAdustment])) + \
                              predictedCursorLoss * self.variableWrapperFunc(torch.FloatTensor([predictedCursorAdustment]))

        self.optimizer.zero_grad()
        totalRebalancedLoss.backward()
        self.optimizer.step()

        totalRewardLoss = float(totalRewardLoss.data.item())
        presentRewardLoss = float(presentRewardLoss.data.item())
        discountedFutureRewardLoss = float(discountedFutureRewardLoss.data.item())
        tracePredictionLoss = float(tracePredictionLoss.data.item())
        predictedExecutionFeaturesLoss = float(predictedExecutionFeaturesLoss.data.item())
        targetHomogenizationLoss = float(targetHomogenizationLoss.data.item())
        predictedCursorLoss = float(predictedCursorLoss.data.item())
        totalLoss = float(totalLoss.data.item())
        totalRebalancedLoss = float(totalRebalancedLoss.data.item())
        batchReward = float(numpy.sum(batch['presentRewards']))

        self.trainingLosses["totalRewardLoss"].append(totalRewardLoss)
        self.trainingLosses["presentRewardLoss"].append(presentRewardLoss)
        self.trainingLosses["discountedFutureRewardLoss"].append(discountedFutureRewardLoss)
        self.trainingLosses["tracePredictionLoss"].append(tracePredictionLoss)
        self.trainingLosses["predictedExecutionFeaturesLoss"].append(predictedExecutionFeaturesLoss)
        self.trainingLosses["targetHomogenizationLoss"].append(targetHomogenizationLoss)
        self.trainingLosses["predictedCursorLoss"].append(predictedCursorLoss)
        self.trainingLosses["totalLoss"].append(totalLoss)
        self.trainingLosses["totalRebalancedLoss"].append(totalRebalancedLoss)

        return totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, targetHomogenizationLoss, predictedCursorLoss, totalLoss, totalRebalancedLoss, batchReward


def processRawImageParallel(rawImage):
    # shrunk = skimage.transform.resize(image, [int(width / 2), int(height / 2)])

    # Convert to HSL representation, but discard the saturation layer
    image = skimage.color.rgb2hsv(rawImage[:, :, :3])
    swapped = numpy.swapaxes(numpy.swapaxes(image, 0, 2), 1, 2)
    hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)
    return hueLightnessImage
