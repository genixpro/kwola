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
        self.batchSize = 20
        self.gamma = 0.50
        self.whichGpu = whichGpu
        self.variableWrapperFunc = (lambda x:x.cuda()) if whichGpu is not None else (lambda x:x)

        self.modelPath = os.path.join(config.getKwolaUserDataDirectory("models"), "deep_learning_model")

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

        self.model = BradNet(self.environment.branchFeatureSize() * 2, len(self.actions), self.environment.branchFeatureSize(), whichGpu=self.whichGpu)

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

            q_values, predictedTrace = self.model({"image": images, "additionalFeature": additionalFeatures})

            actionIndexes = q_values.reshape([-1, width * height * len(self.actionsSorted)]).argmax(1).data

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
            if trace.didActionSucceed == False:
                presentRewards.append(-0.1)
            elif trace.didNewBranchesExecute or trace.isURLNew or trace.isScreenshotNew:
                presentRewards.append(0.2)
            else:
                presentRewards.append(0.0)

        discountRate = 0.95

        # Now compute the discounted reward
        discountedRewards = []
        presentRewards.reverse()
        current = 0
        for reward in presentRewards:
            current *= discountRate
            current += reward
            discountedRewards.append(current)

        discountedRewards.reverse()
        presentRewards.reverse()

        shuffledTraceFrameList = list(zip(executionSession.executionTraces, frames, discountedRewards, presentRewards))
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
            batchDiscountedRewards = []
            batchPresentRewards = []

            for trace, frame, discountedReward, presentReward in batch:
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

                executionTrace = numpy.array(trace.branchExecutionTrace)
                executionTrace = numpy.minimum(executionTrace, numpy.ones_like(executionTrace))

                batchExecutionTraces.append(executionTrace)

                batchDiscountedRewards.append(discountedReward)
                batchPresentRewards.append(presentReward)

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
                "discountedRewards": numpy.array(batchDiscountedRewards, dtype=numpy.float32),
                "presentRewards": numpy.array(batchPresentRewards, dtype=numpy.float32)
            })
            # print("Finished preparing batch #", len(batches), "for", str(executionSession.id))

        # print("Finished preparing all batches for", str(executionSession.id))

        return batches

    def learnFromBatch(self, batch):
        """
            Runs backprop on the neural network with the given batch.

            :param batch: A batch of image/action/output pairs. Should be the return value from prepareBatchesForTestingSequence
            :return:
        """
        width = batch['frames'][0].shape[2]
        height = batch['frames'][0].shape[1]

        q_values, predictedTraces = self.model({
            "image": self.variableWrapperFunc(torch.FloatTensor(numpy.array(batch['frames']))),
            "additionalFeature": self.variableWrapperFunc(torch.FloatTensor(batch['additionalFeatures'])),
            "action_type": batch['actionTypes'],
            "action_x": batch['actionXs'],
            "action_y": batch['actionYs']
        })

        q_values = q_values.reshape([-1, height * width * len(self.actions)])

        action_indexes = self.variableWrapperFunc(torch.LongTensor(numpy.array(batch['actionIndexes'], dtype=numpy.int32)))
        action_indexes = action_indexes.unsqueeze(1)

        q_value = q_values.gather(1, action_indexes).squeeze(1)

        torchBatchDiscountedRewards = self.variableWrapperFunc(torch.FloatTensor(numpy.array(batch['discountedRewards'])))
        #
        # print(q_value)
        # print(q_value.shape)
        # print(torchBatchDiscountedRewards)
        # print(torchBatchDiscountedRewards.shape)

        rewardLoss = (q_value - torchBatchDiscountedRewards).pow(2).mean()

        tracePredictionLoss = (predictedTraces - self.variableWrapperFunc(torch.FloatTensor(batch['executionTraces']))).pow(2).mean()

        totalLoss = rewardLoss + tracePredictionLoss

        self.optimizer.zero_grad()
        totalLoss.backward()
        self.optimizer.step()

        rewardLoss = float(rewardLoss.data.item())
        tracePredictionLoss = float(tracePredictionLoss.data.item())
        totalLoss = float(totalLoss.data.item())
        batchReward = float(numpy.sum(batch['presentRewards']))

        return rewardLoss, tracePredictionLoss, totalLoss, batchReward


def processRawImageParallel(rawImage):
    # shrunk = skimage.transform.resize(image, [int(width / 2), int(height / 2)])

    # Convert to HSL representation, but discard the saturation layer
    image = skimage.color.rgb2hsv(rawImage[:, :, :3])
    swapped = numpy.swapaxes(numpy.swapaxes(image, 0, 2), 1, 2)
    hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)
    return hueLightnessImage
