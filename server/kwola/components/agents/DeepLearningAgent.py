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

    def __init__(self, whichGpu="all"):
        super().__init__()

        self.num_frames = 1400000
        self.batchSize = 14
        self.gamma = 0.50
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

        self.model = BradNet(self.environment.branchFeatureSize() * 2, len(self.actions), self.environment.branchFeatureSize(), 12, len(self.cursors), whichGpu=self.whichGpu)

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
        epsilon = 0.75
        if random.random() > epsilon:
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
                actionType = random.randrange(len(self.actionsSorted))
                actionX = random.randrange(width)
                actionY = random.randrange(height)

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

            if not trace.didActionSucceed:
                tracePresentReward -= 0.02

            if trace.didCodeExecute:
                tracePresentReward += 0.001
            else:
                tracePresentReward -= -0.01

            if trace.didNewBranchesExecute:
                tracePresentReward += 0.30

            if trace.hadNetworkTraffic:
                tracePresentReward += 0.001

            if trace.hadNewNetworkTraffic:
                tracePresentReward += 0.10

            if trace.didScreenshotChange:
                tracePresentReward += 0.001
            else:
                tracePresentReward -= -0.01

            if trace.isScreenshotNew:
                tracePresentReward += 0.005

            if trace.didURLChange:
                tracePresentReward += 0.001

            if trace.isURLNew:
                tracePresentReward += 0.01

            if trace.hadLogOutput:
                tracePresentReward += 0.001

            presentRewards.append(tracePresentReward)

        discountRate = 0.95

        # Now compute the discounted reward
        discountedFutureRewards = []
        presentRewards.reverse()
        current = 0
        for reward in presentRewards:
            current *= discountRate
            discountedFutureRewards.append(current)
            current += reward

        discountedFutureRewards.reverse()
        presentRewards.reverse()

        # Create the decaying future execution trace for the prediction algorithm
        tracesReversed = list(executionSession.executionTraces)
        tracesReversed.reverse()
        currentTrace = numpy.zeros_like(executionSession.executionTraces[0].branchExecutionTrace)
        executionTraceDiscountRate = 0.90
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

        for batch in grouper(self.batchSize, shuffledTraceFrameList):
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
        width = batch['frames'][0].shape[2]
        height = batch['frames'][0].shape[1]

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
            targetHomogenizationLoss = ((pixelFeaturesImageMasked - averageFeatures) * rewardPixelMask).pow(2).sum() / (countPixelMask * self.model.pixelFeatureCount)

            sampleLoss = presentRewardLoss + discountedFutureRewardLoss
            totalRewardLosses.append(sampleLoss.unsqueeze(0))

            presentRewardLosses.append(presentRewardLoss.unsqueeze(0))
            discountedFutureRewardLosses.append(discountedFutureRewardLoss.unsqueeze(0))

            targetHomogenizationLosses.append(targetHomogenizationLoss.unsqueeze(0))

        tracePredictionLoss = (predictedTraces - self.variableWrapperFunc(torch.FloatTensor(batch['executionTraces']))).mean()

        predictedExecutionFeaturesLoss = (predictedExecutionFeatures - self.variableWrapperFunc(torch.FloatTensor(batch['executionFeatures']))).mean()

        predictedCursorLoss = (predictedCursors - self.variableWrapperFunc(torch.FloatTensor(batch['cursors']))).mean()

        totalRewardLoss = torch.mean(torch.cat(totalRewardLosses))
        presentRewardLoss = torch.mean(torch.cat(presentRewardLosses))
        discountedFutureRewardLoss = torch.mean(torch.cat(discountedFutureRewardLosses))
        targetHomogenizationLoss = torch.mean(torch.cat(targetHomogenizationLosses))

        totalLoss = totalRewardLoss + tracePredictionLoss + predictedExecutionFeaturesLoss + targetHomogenizationLoss + predictedCursorLoss

        self.optimizer.zero_grad()
        totalLoss.backward()
        self.optimizer.step()

        totalRewardLoss = float(totalRewardLoss.data.item())
        presentRewardLoss = float(presentRewardLoss.data.item())
        discountedFutureRewardLoss = float(discountedFutureRewardLoss.data.item())
        tracePredictionLoss = float(tracePredictionLoss.data.item())
        predictedExecutionFeaturesLoss = float(predictedExecutionFeaturesLoss.data.item())
        targetHomogenizationLoss = float(targetHomogenizationLoss.data.item())
        predictedCursorLoss = float(predictedCursorLoss.data.item())
        totalLoss = float(totalLoss.data.item())
        batchReward = float(numpy.sum(batch['presentRewards']))

        return totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, targetHomogenizationLoss, predictedCursorLoss, totalLoss, batchReward


def processRawImageParallel(rawImage):
    # shrunk = skimage.transform.resize(image, [int(width / 2), int(height / 2)])

    # Convert to HSL representation, but discard the saturation layer
    image = skimage.color.rgb2hsv(rawImage[:, :, :3])
    swapped = numpy.swapaxes(numpy.swapaxes(image, 0, 2), 1, 2)
    hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)
    return hueLightnessImage
