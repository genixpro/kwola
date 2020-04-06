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


from .BaseAgent import BaseAgent
from .  TraceNet import   TraceNet
from ...components.utilities.debug_plot import showRewardImageDebug
from ...models.ExecutionTraceModel import ExecutionTrace
from ...models.ExecutionSessionModel import ExecutionSession
from ...models.actions.ClickTapAction import ClickTapAction
from ...models.actions.ClickTapAction import ClickTapAction
from ...models.actions.RightClickAction import RightClickAction
from ...models.actions.TypeAction import TypeAction
from ...models.actions.WaitAction import WaitAction
from skimage.segmentation import felzenszwalb, mark_boundaries
import bz2
import concurrent.futures
import cv2
import bson
import cv2
from ...config.config import Configuration
import copy
from datetime import datetime
import itertools
import math, random
import matplotlib as mpl
import matplotlib.pyplot as plt
import traceback
import numpy
import os
import os.path
import os.path
import pandas
import pickle
import scipy.signal
import shutil
import skimage
import skimage.color
import sklearn.preprocessing
import skimage.draw
import skimage.io
import skimage.measure
import skimage.segmentation
import skimage.transform
import subprocess
import tempfile
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def grouper(n, iterable):
    """Chunks an iterable into sublists"""
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk


# noinspection PyUnresolvedReferences
class DeepLearningAgent(BaseAgent):
    """
        This class represents a deep learning agent, which uses reinforcement learning to make the automated testing more effective
    """

    def __init__(self, config, whichGpu="all"):
        super().__init__(config)

        self.config = config
        
        self.whichGpu = whichGpu

        if self.whichGpu == "all":
            self.variableWrapperFunc = lambda t, x: t(x).cuda()
        elif self.whichGpu is None:
            self.variableWrapperFunc = lambda t, x: t(x)
        else:
            self.variableWrapperFunc = lambda t, x: t(x).cuda(device=f"cuda:{self.whichGpu}")

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
            "stateValueLoss": [],
            "advantageLoss": [],
            "actionProbabilityLoss": [],
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
                stateDict = torch.load(self.modelPath, map_location=device)
            elif self.whichGpu != "all":
                device = torch.device(f"cuda:{self.whichGpu}")
                stateDict = torch.load(self.modelPath, map_location=device)
            else:
                stateDict = torch.load(self.modelPath)

            self.model.load_state_dict(stateDict)
            self.targetNetwork.load_state_dict(stateDict)


    def save(self, saveName=""):
        """
            Saves the agent to the db / disk.

            :return:
        """

        if saveName:
            saveName = "_" + saveName

        torch.save(self.model.state_dict(), self.modelPath + saveName)


    def initialize(self, branchFeatureSize):
        """
        Initialize the agent.

        :return:
        """
        if self.whichGpu == "all":
            device_ids = [torch.device(f'cuda:{n}') for n in range(2)]
            output_device = device_ids[0]
        elif self.whichGpu is None:
            device_ids = None
            output_device = None
        else:
            device_ids = [torch.device(f'cuda:{self.whichGpu}')]
            output_device = device_ids[0]

        self.model =   TraceNet(self.config, branchFeatureSize * 2, len(self.actions), branchFeatureSize, 12, len(self.cursors))
        self.targetNetwork =   TraceNet(self.config, branchFeatureSize * 2, len(self.actions), branchFeatureSize, 12, len(self.cursors))

        if self.whichGpu == "all":
            self.model = self.model.cuda()
            self.targetNetwork = self.targetNetwork.cuda()
            self.modelParallel = nn.parallel.DistributedDataParallel(module=self.model, device_ids=device_ids, output_device=output_device)
        elif self.whichGpu is None:
            self.model = self.model.cpu()
            self.targetNetwork = self.targetNetwork.cpu()
            self.modelParallel = self.model
        else:
            self.model = self.model.cuda(device=device_ids[0])
            self.targetNetwork = self.targetNetwork.cuda(device=device_ids[0])
            self.modelParallel = nn.parallel.DistributedDataParallel(module=self.model, device_ids=device_ids, output_device=output_device)

        self.optimizer = optim.Adamax(self.model.parameters(),
                                      lr=self.config['training_learning_rate'],
                                      betas=(self.config['training_gradient_exponential_moving_average_decay'],
                                             self.config['training_gradient_squared_exponential_moving_average_decay'])
                                      )

    def updateTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.model.state_dict())

    def processImages(self, images):
        convertedImageFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for image in images:
                convertedImageFuture = executor.submit(DeepLearningAgent.processRawImageParallel, image, self.config)
                convertedImageFutures.append(convertedImageFuture)

        convertedProcessedImages = [
            convertedImageFuture.result() for convertedImageFuture in convertedImageFutures
        ]

        return numpy.array(convertedProcessedImages)


    def createPixelActionMap(self, actionMaps, height, width):
        pixelActionMap = numpy.zeros([len(self.actionsSorted), height, width], dtype=numpy.uint8)

        for element in actionMaps:
            actionTypes = []

            if element['canClick']:
                actionTypes.append(self.actionsSorted.index("click"))
                # actionTypes.append(self.actionsSorted.index("double_click"))
            if element['canType']:
                actionTypes.append(self.actionsSorted.index("typeEmail"))
                actionTypes.append(self.actionsSorted.index("typePassword"))
                # actionTypes.append(self.actionsSorted.index("typeName"))
                # actionTypes.append(self.actionsSorted.index("typeNumber"))
                # actionTypes.append(self.actionsSorted.index("typeBrackets"))
                # actionTypes.append(self.actionsSorted.index("typeMath"))
                # actionTypes.append(self.actionsSorted.index("typeOtherSymbol"))
                # actionTypes.append(self.actionsSorted.index("typeParagraph"))
                actionTypes.append(self.actionsSorted.index("clear"))

            for actionTypeIndex in actionTypes:
                pixelActionMap[actionTypeIndex, int(element['top'] * self.config['model_image_downscale_ratio'])
                                                :int(element['bottom'] * self.config['model_image_downscale_ratio']),
                                                int(element['left'] * self.config['model_image_downscale_ratio'])
                                                :int(element['right'] * self.config['model_image_downscale_ratio'])] = 1

        return pixelActionMap

    def getActionMapsIntersectingWithAction(self, action, actionMaps):
        selected = []
        for actionMap in actionMaps:
            if actionMap.left <= action.x <= actionMap.right and actionMap.top <= action.y <= actionMap.bottom:
                selected.append(actionMap)

        return selected

    def nextBestActions(self, stepNumber, rawImages, envActionMaps, additionalFeatures, recentActions, shouldBeRandom=False):
        """
            Return the next best action predicted by the agent.

            :return:
        """
        processedImages = self.processImages(rawImages)
        actions = []

        width = processedImages.shape[3]
        height = processedImages.shape[2]

        batchSampleIndexes = []
        imageBatch = []
        additionalFeatureVectorBatch = []
        pixelActionMapsBatch = []
        epsilonsPerSample = []
        recentActionsBatch = []
        actionMapsBatch = []
        recentActionsCountsBatch = []

        modelDownscale = self.config['model_image_downscale_ratio']

        for sampleIndex, image, additionalFeatureVector, sampleActionMaps, sampleRecentActions in zip(range(len(processedImages)), processedImages, additionalFeatures, envActionMaps, recentActions):
            randomActionProbability = (float(sampleIndex + 1) / float(len(processedImages))) * 0.50 * (1 + (stepNumber / self.config['testing_sequence_length']))
            weightedRandomActionProbability = (float(sampleIndex + 1) / float(len(processedImages))) * 0.50 * (1 + (stepNumber / self.config['testing_sequence_length']))

            filteredSampleActionMaps = []
            filteredSampleActionRecentActionCounts = []
            for map in sampleActionMaps:
                # Check to see if the map is out of the screen
                if (map.top * modelDownscale) > height or (map.bottom * modelDownscale) < 0 or (map.left * modelDownscale) > width or (map.right * modelDownscale) < 0:
                    # skip this action map, don't add it to the filtered list
                    continue

                count = 0
                for recentAction in sampleRecentActions:
                    for recentActionMap in self.getActionMapsIntersectingWithAction(recentAction, recentAction.actionMapsAvailable):
                        if map.doesOverlapWith(recentActionMap, tolerancePixels=self.config['testing_repeat_action_pixel_overlap_tolerance']):
                            count += 1
                            break
                if count < self.config['testing_max_repeat_action_maps_without_new_branches']:
                    filteredSampleActionMaps.append(map)
                    filteredSampleActionRecentActionCounts.append(count)

            if len(filteredSampleActionMaps) > 0:
                sampleActionMaps = filteredSampleActionMaps
                sampleActionRecentActionCounts = filteredSampleActionRecentActionCounts
            else:
                sampleActionRecentActionCounts = [1] * len(sampleActionMaps)

            pixelActionMap = self.createPixelActionMap(sampleActionMaps, height, width)

            if random.random() > randomActionProbability and not shouldBeRandom:
                batchSampleIndexes.append(sampleIndex)
                imageBatch.append(image)
                additionalFeatureVectorBatch.append(additionalFeatureVector)
                actionMapsBatch.append(sampleActionMaps)
                pixelActionMapsBatch.append(pixelActionMap)
                recentActionsBatch.append(sampleRecentActions)
                epsilonsPerSample.append(weightedRandomActionProbability)
                recentActionsCountsBatch.append(sampleActionRecentActionCounts)
            else:
                actionX, actionY, actionType = self.getRandomAction(sampleActionRecentActionCounts, sampleActionMaps, pixelActionMap)

                action = self.actions[self.actionsSorted[actionType]](
                            int(actionX / modelDownscale),
                            int(actionY / modelDownscale)
                )
                action.source = "random"
                action.predictedReward = None
                action.wasRepeatOverride = False
                action.actionMapsAvailable = sampleActionMaps
                actions.append((sampleIndex, action))

        if len(imageBatch) > 0:
            imageTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(imageBatch))
            additionalFeatureTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(additionalFeatureVectorBatch))
            pixelActionMapTensor = self.variableWrapperFunc(torch.FloatTensor, pixelActionMapsBatch)
            stepNumberTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array([stepNumber] * len(imageBatch)))

            with torch.no_grad():
                self.model.eval()

                outputs = self.modelParallel({
                    "image": imageTensor,
                    "additionalFeature": additionalFeatureTensor,
                    "pixelActionMaps": pixelActionMapTensor,
                    "stepNumber": stepNumberTensor,
                    "outputStamp": False,
                    "computeExtras": False,
                    "computeRewards": False,
                    "computeActionProbabilities": True,
                    "computeStateValues": False,
                    "computeAdvantageValues": True
                })

                actionProbabilities = outputs['actionProbabilities'].cpu()
                # actionProbabilities = outputs['advantage'].cpu()

            for sampleIndex, sampleEpsilon, sampleActionProbs, sampleRecentActions, sampleActionMaps, sampleActionRecentActionCounts, samplePixelActionMap in zip(batchSampleIndexes, epsilonsPerSample, actionProbabilities, recentActionsBatch, actionMapsBatch, recentActionsCountsBatch, pixelActionMapsBatch):
                weighted = bool(random.random() < sampleEpsilon)
                override = False
                source = None
                actionType = None
                actionX = None
                actionY = None
                samplePredictedReward = None
                if not weighted:
                    source = "prediction"

                    actionX, actionY, actionType = self.getActionInfoTensorsFromRewardMap(sampleActionProbs)
                    actionX = actionX.data.item()
                    actionY = actionY.data.item()
                    actionType = actionType.data.item()
                    
                    samplePredictedReward = sampleActionProbs[actionType, actionY, actionX].data.item()

                    potentialAction = self.actions[self.actionsSorted[actionType]](int(actionX / self.config['model_image_downscale_ratio']), int(actionY / self.config['model_image_downscale_ratio']))
                    potentialActionMaps = self.getActionMapsIntersectingWithAction(potentialAction, sampleActionMaps)

                    # If the network is predicting the same action as it did within the recent turns list, down to the exact pixels
                    # of the action maps, that usually implies its stuck and its action had no effect on the environment. Switch
                    # to random weighted to try and break out of this stuck condition. The recent action list is reset every
                    # time the algorithm discovers new code branches, e.g. new functionality so this helps ensure the algorithm
                    # stays exploring instead of getting stuck but can learn different behaviours with the same elements
                    for recentAction in sampleRecentActions:
                        recentActionMaps = self.getActionMapsIntersectingWithAction(recentAction, recentAction.actionMapsAvailable)

                        if recentAction.type != potentialAction.type:
                            continue

                        allEqual = True
                        for recentMap in recentActionMaps:
                            found = False
                            for potentialMap in potentialActionMaps:
                                if recentMap.doesOverlapWith(potentialMap, tolerancePixels=self.config['testing_repeat_action_pixel_overlap_tolerance']):
                                    found = True
                                    break
                            if not found:
                                allEqual = False
                                break

                        if allEqual:
                            weighted = True
                            override = True
                            break

                if weighted:
                    reshaped = numpy.array(sampleActionProbs.data).reshape([len(self.actionsSorted) * height * width])
                    reshapedSum = numpy.sum(reshaped)
                    if reshapedSum > 0:
                        reshaped = reshaped / reshapedSum

                    try:
                        actionIndex = numpy.random.choice(range(len(self.actionsSorted) * height * width), p=reshaped)

                        newProbs = numpy.zeros([len(self.actionsSorted) * height * width])
                        newProbs[actionIndex] = 1

                        newProbs = newProbs.reshape([len(self.actionsSorted), height * width])
                        actionType = newProbs.max(axis=1).argmax(axis=0)
                        newProbs = newProbs.reshape([len(self.actionsSorted), height, width])
                        actionY = newProbs[actionType].max(axis=1).argmax(axis=0)
                        actionX = newProbs[actionType, actionY].argmax(axis=0)

                        source = "weighted_random"
                        samplePredictedReward = sampleActionProbs[actionType, actionY, actionX].data.item()
                    except ValueError:
                        print(datetime.now(), "Error in weighted random choice! Probabilities do not all add up to 1. Picking a random action.", flush=True)
                        # This usually occurs when all the probabilities do not add up to 1, due to floating point error.
                        # So instead we just pick an action randomly.
                        actionX, actionY, actionType = self.getRandomAction(sampleActionRecentActionCounts, sampleActionMaps, samplePixelActionMap)
                        source = "random"

                action = self.actions[self.actionsSorted[actionType]](int(actionX / self.config['model_image_downscale_ratio']), int(actionY / self.config['model_image_downscale_ratio']))
                action.source = source
                action.predictedReward = samplePredictedReward
                action.actionMapsAvailable = sampleActionMaps
                action.wasRepeatOverride = override
                actions.append((sampleIndex, action))

        sortedActions = sorted(actions, key=lambda row: row[0])

        return [action[1] for action in sortedActions]

    def getRandomAction(self, sampleActionRecentActionCounts, sampleActionMaps, pixelActionMap):
        width = pixelActionMap.shape[2]
        height = pixelActionMap.shape[1]

        actionMapWeights = numpy.array([self.elementBaseWeights.get(map.elementType, self.elementBaseWeights['other']) for map in sampleActionMaps]) / (numpy.array(sampleActionRecentActionCounts) + 1)

        chosenActionMapIndex = numpy.random.choice(range(len(sampleActionMaps)), p=scipy.special.softmax(actionMapWeights))
        chosenActionMap = sampleActionMaps[chosenActionMapIndex]

        actionX = random.randint(max(0, int(min(width - 1, chosenActionMap.left * self.config['model_image_downscale_ratio']))),
                                 max(0, int(min(chosenActionMap.right * self.config['model_image_downscale_ratio'] - 1, width - 1))))
        actionY = random.randint(max(0, int(min(height - 1, chosenActionMap.top * self.config['model_image_downscale_ratio']))),
                                 max(0, int(min(chosenActionMap.bottom * self.config['model_image_downscale_ratio'] - 1, height - 1))))

        possibleActionsAtPixel = pixelActionMap[:, actionY, actionX]
        possibleActionIndexes = [actionIndex for actionIndex in range(len(self.actionsSorted)) if possibleActionsAtPixel[actionIndex]]
        possibleActionWeights = [self.actionBaseWeights[actionIndex] for actionIndex in possibleActionIndexes]

        possibleActionBoosts = []
        for actionIndex in possibleActionIndexes:
            boostKeywords = self.actionProbabilityBoostKeywords[actionIndex]

            boost = False
            for keyword in boostKeywords:
                if keyword in chosenActionMap.keywords:
                    boost = True
                    break

            if boost:
                possibleActionBoosts.append(1.5)
            else:
                possibleActionBoosts.append(1.0)

        try:
            actionType = numpy.random.choice(possibleActionIndexes, p=scipy.special.softmax(numpy.array(possibleActionWeights) * numpy.array(possibleActionBoosts)))
        except ValueError:
            actionType = random.choice(range(len(self.actionsSorted)))

        return actionX, actionY, actionType

    @staticmethod
    def computePresentRewards(executionTraces, config):

        # First compute the present reward at each time step
        presentRewards = []
        for trace in executionTraces:
            tracePresentReward = 0.0

            if trace.didActionSucceed:
                tracePresentReward += config['reward_action_success']
            else:
                tracePresentReward += config['reward_action_failure']

            if trace.didCodeExecute:
                tracePresentReward += config['reward_code_executed']
            else:
                tracePresentReward += config['reward_no_code_executed']

            if trace.didNewBranchesExecute:
                tracePresentReward += config['reward_new_code_executed']
            else:
                tracePresentReward += config['reward_no_new_code_executed']

            if trace.hadNetworkTraffic:
                tracePresentReward += config['reward_network_traffic']
            else:
                tracePresentReward += config['reward_no_network_traffic']

            if trace.hadNewNetworkTraffic:
                tracePresentReward += config['reward_new_network_traffic']
            else:
                tracePresentReward += config['reward_no_new_network_traffic']

            if trace.didScreenshotChange:
                tracePresentReward += config['reward_screenshot_changed']
            else:
                tracePresentReward += config['reward_no_screenshot_change']

            if trace.isScreenshotNew:
                tracePresentReward += config['reward_new_screenshot']
            else:
                tracePresentReward += config['reward_no_new_screenshot']

            if trace.didURLChange:
                tracePresentReward += config['reward_url_changed']
            else:
                tracePresentReward += config['reward_no_url_change']

            if trace.isURLNew:
                tracePresentReward += config['reward_new_url']
            else:
                tracePresentReward += config['reward_no_new_url']

            if trace.hadLogOutput:
                tracePresentReward += config['reward_log_output']
            else:
                tracePresentReward += config['reward_no_log_output']

            presentRewards.append(tracePresentReward)
        return presentRewards

    @staticmethod
    def computeDiscountedFutureRewards(executionTraces, config):
        # First compute the present reward at each time step
        presentRewards = DeepLearningAgent.computePresentRewards(executionTraces, config)

        # Now compute the discounted reward
        discountedFutureRewards = []
        presentRewards.reverse()
        current = 0
        for reward in presentRewards:
            current *= config['reward_discount_rate']
            discountedFutureRewards.append(current)
            current += reward

        discountedFutureRewards.reverse()

        return discountedFutureRewards

    @staticmethod
    def readVideoFrames(videoFilePath):
        cap = cv2.VideoCapture(videoFilePath)

        rawImages = []

        while (cap.isOpened()):
            ret, rawImage = cap.read()
            if ret:
                rawImage = numpy.flip(rawImage, axis=2) # OpenCV reads everything in BGR format for some reason so flip to RGB
                rawImages.append(rawImage)
            else:
                break

        return rawImages


    def getActionInfoTensorsFromRewardMap(self, rewardMapTensor):
        width = rewardMapTensor.shape[2]
        height = rewardMapTensor.shape[1]

        actionType = rewardMapTensor.reshape([len(self.actionsSorted), width * height]).max(dim=1)[0].argmax(0)
        actionY = rewardMapTensor[actionType].max(dim=1)[0].argmax(0)
        actionX = rewardMapTensor[actionType, actionY].argmax(0)
        
        return actionX, actionY, actionType


    def createDebugVideoForExecutionSession(self, executionSession):
        videoPath = self.config.getKwolaUserDataDirectory("videos")

        rawImages = DeepLearningAgent.readVideoFrames(os.path.join(videoPath, f"{str(executionSession.id)}.mp4"))

        executionTraces = [ExecutionTrace.loadFromDisk(traceId, self.config) for traceId in executionSession.executionTraces]

        # Filter out any traces that failed to load. Generally this only happens when you interrupt the process
        # while it is writing a file. So it happens to devs but not in production. Still we protect against
        # this case in several places throughout the code.
        executionTracesFiltered = [trace for trace in executionTraces if trace is not None]

        presentRewards = DeepLearningAgent.computePresentRewards(executionTracesFiltered, self.config)

        discountedFutureRewards = DeepLearningAgent.computeDiscountedFutureRewards(executionTracesFiltered, self.config)

        tempScreenshotDirectory = tempfile.mkdtemp()

        debugImageIndex = 0

        lastRawImage = rawImages.pop(0)

        mpl.use('Agg')
        mpl.rcParams['figure.max_open_warning'] = 1000

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for trace, rawImage in zip(executionTraces, rawImages):
                if trace is not None:
                    future = executor.submit(self.createDebugImagesForExecutionTrace, str(executionSession.id), debugImageIndex, trace.to_json(), rawImage, lastRawImage, presentRewards, discountedFutureRewards, tempScreenshotDirectory)
                    futures.append(future)

                    debugImageIndex += 2
                    lastRawImage = rawImage

            concurrent.futures.wait(futures)

        subprocess.run(['ffmpeg', '-f', 'image2', "-r", "2", '-i', 'kwola-screenshot-%05d.png', '-vcodec', 'libx264', '-crf', '15', "debug.mp4"], cwd=tempScreenshotDirectory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        moviePath = os.path.join(tempScreenshotDirectory, "debug.mp4")

        with open(moviePath, "rb") as file:
            videoData = file.read()

        shutil.rmtree(tempScreenshotDirectory)

        return videoData

    def createDebugImagesForExecutionTrace(self, executionSessionId, debugImageIndex, trace, rawImage, lastRawImage, presentRewards, discountedFutureRewards, tempScreenshotDirectory):
        try:
            trace = ExecutionTrace.from_json(trace)

            topSize = 250
            bottomSize = 250
            leftSize = 100
            rightSize = 1250
            topMargin = 25

            imageHeight = rawImage.shape[0]
            imageWidth = rawImage.shape[1]

            presentReward = presentRewards[trace.frameNumber - 1]
            discountedFutureReward = discountedFutureRewards[trace.frameNumber - 1]

            def addDebugCircleToImage(image, trace):
                targetCircleCoordsRadius30 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                           int(leftSize + trace.actionPerformed.x), 30,
                                                                           shape=[int(imageWidth + extraWidth),
                                                                                  int(imageHeight + extraHeight)])
                targetCircleCoordsRadius20 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                           int(leftSize + trace.actionPerformed.x), 20,
                                                                           shape=[int(imageWidth + extraWidth),
                                                                                  int(imageHeight + extraHeight)])
                targetCircleCoordsRadius10 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                           int(leftSize + trace.actionPerformed.x), 10,
                                                                           shape=[int(imageWidth + extraWidth),
                                                                                  int(imageHeight + extraHeight)])
                targetCircleCoordsRadius5 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y),
                                                                          int(leftSize + trace.actionPerformed.x), 5,
                                                                          shape=[int(imageWidth + extraWidth),
                                                                                 int(imageHeight + extraHeight)])
                image[targetCircleCoordsRadius30] = [255, 0, 0]
                image[targetCircleCoordsRadius20] = [255, 0, 0]
                image[targetCircleCoordsRadius10] = [255, 0, 0]
                image[targetCircleCoordsRadius5] = [255, 0, 0]

            def addCropViewToImage(image, trace):
                cropLeft, cropTop, cropRight, cropBottom = self.calculateTrainingCropPosition(trace.actionPerformed.x, trace.actionPerformed.y, imageWidth, imageHeight)

                cropRectangle = skimage.draw.rectangle_perimeter((int(topSize + cropTop), int(leftSize + cropLeft)), (int(topSize + cropBottom), int(leftSize + cropRight)))
                image[cropRectangle] = [255, 0, 0]

            def addDebugTextToImage(image, trace):
                fontSize = 0.5
                fontThickness = 1
                fontColor = (0, 0, 0)

                columnOneLeft = leftSize
                columnTwoLeft = leftSize + 300
                columnThreeLeft = leftSize + 550
                lineOneTop = topMargin + 20
                lineTwoTop = topMargin + 40
                lineThreeTop = topMargin + 60
                lineFourTop = topMargin + 80
                lineFiveTop = topMargin + 100
                lineSixTop = topMargin + 120
                lineSevenTop = topMargin + 140
                lineEightTop = topMargin + 160
                lineNineTop = topMargin + 180

                cv2.putText(image, f"URL {trace.startURL}", (columnOneLeft, lineOneTop), cv2.FONT_HERSHEY_SIMPLEX, fontSize,
                            fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"{str(executionSessionId)}", (columnOneLeft, lineTwoTop), cv2.FONT_HERSHEY_SIMPLEX,
                            fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"Frame {trace.frameNumber}", (columnOneLeft, lineThreeTop), cv2.FONT_HERSHEY_SIMPLEX,
                            fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image,
                            f"Action {trace.actionPerformed.type} at {trace.actionPerformed.x},{trace.actionPerformed.y}",
                            (columnOneLeft, lineFourTop), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness,
                            cv2.LINE_AA)
                cv2.putText(image, f"Source: {str(trace.actionPerformed.source)}", (columnOneLeft, lineFiveTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Succeed: {str(trace.didActionSucceed)}", (columnOneLeft, lineSixTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"Error: {str(trace.didErrorOccur)}", (columnOneLeft, lineSevenTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"New Error: {str(trace.didNewErrorOccur)}", (columnOneLeft, lineEightTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"Override: {str(trace.actionPerformed.wasRepeatOverride)}", (columnOneLeft, lineNineTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Code Execute: {str(trace.didCodeExecute)}", (columnTwoLeft, lineTwoTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"New Branches: {str(trace.didNewBranchesExecute)}", (columnTwoLeft, lineThreeTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Network Traffic: {str(trace.hadNetworkTraffic)}", (columnTwoLeft, lineFourTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"New Network Traffic: {str(trace.hadNewNetworkTraffic)}", (columnTwoLeft, lineFiveTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Screenshot Change: {str(trace.didScreenshotChange)}", (columnTwoLeft, lineSixTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"New Screenshot: {str(trace.isScreenshotNew)}", (columnTwoLeft, lineSevenTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Cursor: {str(trace.cursor)}", (columnTwoLeft, lineEightTop), cv2.FONT_HERSHEY_SIMPLEX,
                            fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Discounted Future Reward: {(discountedFutureReward):.3f}",
                            (columnThreeLeft, lineTwoTop), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness,
                            cv2.LINE_AA)
                cv2.putText(image, f"Present Reward: {(presentReward):.3f}", (columnThreeLeft, lineThreeTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Branch Coverage: {(trace.cumulativeBranchCoverage * 100):.2f}%",
                            (columnThreeLeft, lineFourTop), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness,
                            cv2.LINE_AA)

                cv2.putText(image, f"URL Change: {str(trace.didURLChange)}", (columnThreeLeft, lineFiveTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
                cv2.putText(image, f"New URL: {str(trace.isURLNew)}", (columnThreeLeft, lineSixTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                cv2.putText(image, f"Had Log Output: {trace.hadLogOutput}", (columnThreeLeft, lineSevenTop),
                            cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

                if trace.actionPerformed.predictedReward:
                    cv2.putText(image, f"Predicted Reward: {(trace.actionPerformed.predictedReward):.3f}", (columnThreeLeft, lineEightTop),
                                cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)


            def addBottomRewardChartToImage(image, trace):
                rewardChartFigure = plt.figure(figsize=(imageWidth / 100, (bottomSize - 50) / 100), dpi=100)
                rewardChartAxes = rewardChartFigure.add_subplot(111)

                xCoords = numpy.array(range(len(presentRewards)))

                rewardChartAxes.set_ylim(ymin=-2.0, ymax=15.0)

                rewardChartAxes.plot(xCoords, numpy.array(presentRewards) + numpy.array(discountedFutureRewards))

                rewardChartAxes.set_xticks(range(0, len(presentRewards), 5))
                rewardChartAxes.set_xticklabels([str(n) for n in range(0, len(presentRewards), 5)])
                rewardChartAxes.set_yticks(numpy.arange(0, 1, 1.0))
                rewardChartAxes.set_yticklabels(["" for n in range(2)])
                rewardChartAxes.set_title("Net Present Reward")

                # ax.grid()
                rewardChartFigure.tight_layout()

                rewardChartAxes.set_xlim(xmin=trace.frameNumber - 20, xmax=trace.frameNumber + 20)
                vline = rewardChartAxes.axvline(trace.frameNumber - 1, color='black', linewidth=2)
                hline = rewardChartAxes.axhline(0, color='grey', linewidth=1)

                # If we haven't already shown or saved the plot, then we need to
                # draw the figure first...
                rewardChartFigure.canvas.draw()

                # Now we can save it to a numpy array.
                rewardChart = numpy.fromstring(rewardChartFigure.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
                rewardChart = rewardChart.reshape(rewardChartFigure.canvas.get_width_height()[::-1] + (3,))

                image[topSize + imageHeight:-50, leftSize:-rightSize] = rewardChart

                vline.remove()
                hline.remove()
                plt.close(rewardChartFigure)

            def addRightSideDebugCharts(plotImage, rawImage, trace):
                chartTopMargin = 75
                numColumns = 4
                numRows = 4

                mainColorMap = plt.get_cmap('inferno')
                greyColorMap = plt.get_cmap('gray')

                currentFig = 1

                mainFigure = plt.figure(
                    figsize=((rightSize) / 100, (imageHeight + bottomSize + topSize - chartTopMargin) / 100), dpi=100)

                rewardPredictionAxes = [
                    mainFigure.add_subplot(numColumns, numRows, actionIndex + currentFig)
                    for actionIndex, action in enumerate(self.actionsSorted)
                ]

                currentFig += len(rewardPredictionAxes)

                advantagePredictionAxes = [
                    mainFigure.add_subplot(numColumns, numRows, actionIndex + currentFig)
                    for actionIndex, action in enumerate(self.actionsSorted)
                ]

                currentFig += len(advantagePredictionAxes)

                actionProbabilityPredictionAxes = [
                    mainFigure.add_subplot(numColumns, numRows, actionIndex + currentFig)
                    for actionIndex, action in enumerate(self.actionsSorted)
                ]

                currentFig += len(actionProbabilityPredictionAxes)

                stampAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                currentFig += 1

                stateValueAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                currentFig += 1

                processedImage = DeepLearningAgent.processRawImageParallel(rawImage, self.config)

                rewardPixelMaskAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                currentFig += 1
                rewardPixelMask = self.createRewardPixelMask(processedImage,
                                                             int(trace.actionPerformed.x * self.config['model_image_downscale_ratio']),
                                                             int(trace.actionPerformed.y * self.config['model_image_downscale_ratio'])
                                                             )
                rewardPixelCount = numpy.count_nonzero(rewardPixelMask)
                rewardPixelMaskAxes.imshow(rewardPixelMask, vmin=0, vmax=1, cmap=plt.get_cmap("gray"), interpolation="bilinear")
                rewardPixelMaskAxes.set_xticks([])
                rewardPixelMaskAxes.set_yticks([])
                rewardPixelMaskAxes.set_title(f"{rewardPixelCount} target pixels")

                # pixelActionMapAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                # currentFig += 1
                pixelActionMap = self.createPixelActionMap(trace.actionPerformed.actionMapsAvailable, processedImage.shape[1],  processedImage.shape[2])
                # actionPixelCount = numpy.count_nonzero(pixelActionMap)
                # pixelActionMapAxes.imshow(numpy.swapaxes(numpy.swapaxes(pixelActionMap, 0, 1), 1, 2) * 255, interpolation="bilinear")
                # pixelActionMapAxes.set_xticks([])
                # pixelActionMapAxes.set_yticks([])
                # pixelActionMapAxes.set_title(f"{actionPixelCount} action pixels")

                additionalFeature = self.prepareAdditionalFeaturesForTrace(trace)

                with torch.no_grad():
                    self.model.eval()
                    outputs = \
                        self.modelParallel({"image": self.variableWrapperFunc(torch.FloatTensor, numpy.array([processedImage])),
                                    "additionalFeature": self.variableWrapperFunc(torch.FloatTensor, [additionalFeature]),
                                    "pixelActionMaps": self.variableWrapperFunc(torch.FloatTensor, numpy.array([pixelActionMap])),
                                    "stepNumber": self.variableWrapperFunc(torch.FloatTensor, numpy.array([trace.frameNumber - 1])),
                                    "outputStamp": True,
                                    "computeExtras": False,
                                    "computeRewards": True,
                                    "computeActionProbabilities": True,
                                    "computeStateValues": True,
                                    "computeAdvantageValues": True
                        })

                    totalRewardPredictions = numpy.array((outputs['presentRewards'] + outputs['discountFutureRewards']).data)
                    # stateValuePredictions = numpy.array((outputs['stateValues']).data)
                    advantagePredictions = numpy.array((outputs['advantage']).data)
                    actionProbabilities = numpy.array((outputs['actionProbabilities']).data)
                    stateValue = numpy.array((outputs['stateValues'][0]).data)
                    stamp = outputs['stamp']




                for actionIndex, action in enumerate(self.actionsSorted):
                    actionY = actionProbabilities[0][actionIndex].max(axis=1).argmax(axis=0)
                    actionX = actionProbabilities[0][actionIndex, actionY].argmax(axis=0)

                    actionX = int(actionX / self.config["model_image_downscale_ratio"])
                    actionY = int(actionY / self.config["model_image_downscale_ratio"])

                    targetCircleCoordsRadius10 = skimage.draw.circle_perimeter(int(topSize + actionY),
                                                                              int(leftSize + actionX), 10,
                                                                              shape=[int(imageWidth + extraWidth),
                                                                                     int(imageHeight + extraHeight)])

                    targetCircleCoordsRadius5 = skimage.draw.circle_perimeter(int(topSize + actionY),
                                                                              int(leftSize + actionX), 5,
                                                                              shape=[int(imageWidth + extraWidth),
                                                                                     int(imageHeight + extraHeight)])
                    plotImage[targetCircleCoordsRadius10] = [0, 0, 255]
                    plotImage[targetCircleCoordsRadius5] = [0, 0, 255]


                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(totalRewardPredictions[0][actionIndex]))

                    rewardPredictionAxes[actionIndex].set_xticks([])
                    rewardPredictionAxes[actionIndex].set_yticks([])

                    rewardPredictionsShrunk = skimage.measure.block_reduce(totalRewardPredictions[0][actionIndex], (4, 4), numpy.max)

                    im = rewardPredictionAxes[actionIndex].imshow(rewardPredictionsShrunk, cmap=mainColorMap, interpolation="nearest", vmin=-2, vmax=10)
                    rewardPredictionAxes[actionIndex].set_title(f"{action} {maxValue:.2f} reward")
                    mainFigure.colorbar(im, ax=rewardPredictionAxes[actionIndex], orientation='vertical')

                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(advantagePredictions[0][actionIndex]))

                    advantagePredictionAxes[actionIndex].set_xticks([])
                    advantagePredictionAxes[actionIndex].set_yticks([])

                    advanagePredictionsShrunk = skimage.measure.block_reduce(advantagePredictions[0][actionIndex], (4, 4), numpy.max)

                    im = advantagePredictionAxes[actionIndex].imshow(advanagePredictionsShrunk, cmap=mainColorMap, interpolation="nearest", vmin=-1, vmax=1)
                    advantagePredictionAxes[actionIndex].set_title(f"{action} {maxValue:.2f} advantage")
                    mainFigure.colorbar(im, ax=advantagePredictionAxes[actionIndex], orientation='vertical')

                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(actionProbabilities[0][actionIndex]))

                    actionProbabilityPredictionAxes[actionIndex].set_xticks([])
                    actionProbabilityPredictionAxes[actionIndex].set_yticks([])

                    actionProbabilityPredictionsShrunk = skimage.measure.block_reduce(actionProbabilities[0][actionIndex], (4, 4), numpy.max)

                    im = actionProbabilityPredictionAxes[actionIndex].imshow(actionProbabilityPredictionsShrunk, cmap=mainColorMap, interpolation="nearest")
                    actionProbabilityPredictionAxes[actionIndex].set_title(f"{action} {maxValue:.3f} prob")
                    mainFigure.colorbar(im, ax=actionProbabilityPredictionAxes[actionIndex], orientation='vertical')

                stampAxes.set_xticks([])
                stampAxes.set_yticks([])
                stampImageWidth = self.config['additional_features_stamp_edge_size'] * self.config['additional_features_stamp_edge_size']
                stampImageHeight = self.config['additional_features_stamp_depth_size']

                stampIm = stampAxes.imshow(numpy.array(stamp.data[0]).reshape([stampImageWidth, stampImageHeight]), cmap=greyColorMap, interpolation="nearest", vmin=-1.0, vmax=5.0)
                mainFigure.colorbar(stampIm, ax=stampAxes, orientation='vertical')
                stampAxes.set_title("Memory Stamp")

                stateValueAxes.set_xticks([])
                stateValueAxes.set_yticks([])
                stateValueIm = stateValueAxes.imshow([stateValue], cmap=mainColorMap, interpolation="nearest", vmin=-2.0, vmax=10.0)
                mainFigure.colorbar(stateValueIm, ax=stateValueAxes, orientation='vertical')
                stateValueAxes.set_title(f"State Value {float(stateValue[0]):.3f}")

                # ax.grid()
                mainFigure.tight_layout()
                mainFigure.canvas.draw()

                # Now we can save it to a numpy array and paste it into the image
                mainChart = numpy.fromstring(mainFigure.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
                mainChart = mainChart.reshape(mainFigure.canvas.get_width_height()[::-1] + (3,))
                plotImage[chartTopMargin:, (-rightSize):] = mainChart
                plt.close(mainFigure)


            extraWidth = leftSize + rightSize
            extraHeight = topSize + bottomSize

            newImage = numpy.ones([imageHeight + extraHeight, imageWidth + extraWidth, 3]) * 255
            newImage[topSize:-bottomSize, leftSize:-rightSize] = lastRawImage
            addDebugTextToImage(newImage, trace)
            addDebugCircleToImage(newImage, trace)
            addCropViewToImage(newImage, trace)
            addBottomRewardChartToImage(newImage, trace)
            addRightSideDebugCharts(newImage, lastRawImage, trace)

            fileName = f"kwola-screenshot-{debugImageIndex:05d}.png"
            filePath = os.path.join(tempScreenshotDirectory, fileName)
            skimage.io.imsave(filePath, numpy.array(newImage, dtype=numpy.uint8))

            newImage = numpy.ones([imageHeight + extraHeight, imageWidth + extraWidth, 3]) * 255
            addDebugTextToImage(newImage, trace)

            newImage[topSize:-bottomSize, leftSize:-rightSize] = rawImage
            addBottomRewardChartToImage(newImage, trace)
            addRightSideDebugCharts(newImage, lastRawImage, trace)

            fileName = f"kwola-screenshot-{debugImageIndex+1:05d}.png"
            filePath = os.path.join(tempScreenshotDirectory, fileName)
            skimage.io.imsave(filePath, numpy.array(newImage, dtype=numpy.uint8))

            print(datetime.now(), "Completed debug image", fileName, flush=True)
        except Exception:
            traceback.print_exc()
            print(datetime.now(), "Failed to create debug image!", flush=True)


    def createRewardPixelMask(self, processedImage, x, y):
        # We use flood-segmentation on the original image to select which pixels we will update reward values for.
        # This works great on UIs because the elements always have big areas of solid-color which respond in the same
        # way.
        rewardPixelMask = skimage.segmentation.flood(numpy.array(processedImage[0], dtype=numpy.float32), (int(y), int(x)))

        return rewardPixelMask


    def prepareAdditionalFeaturesForTrace(self, trace):
        branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
        decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
        additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

        return additionalFeature


    def calculateTrainingCropPosition(self, centerX, centerY, imageWidth, imageHeight, nextStepCrop=False):
        cropWidth = self.config['training_crop_width']
        cropHeight = self.config['training_crop_height']
        if nextStepCrop:
            cropWidth = self.config['training_next_step_crop_width']
            cropHeight = self.config['training_next_step_crop_height']

        cropLeft = centerX - cropWidth / 2
        cropTop = centerY - cropHeight / 2

        cropLeft = max(0, cropLeft)
        cropLeft = min(imageWidth - cropWidth, cropLeft)
        cropLeft = int(cropLeft)

        cropTop = max(0, cropTop)
        cropTop = min(imageHeight - cropHeight, cropTop)
        cropTop = int(cropTop)

        cropRight = int(cropLeft + cropWidth)
        cropBottom = int(cropTop + cropHeight)

        return (cropLeft, cropTop, cropRight, cropBottom)


    @staticmethod
    def computeTotalRewardsParallel(execusionSessionId, configDir):
        config = Configuration(configDir)

        session = ExecutionSession.loadFromDisk(execusionSessionId, config)

        executionTraces = [ExecutionTrace.loadFromDisk(traceId, config, omitLargeFields=True) for traceId in session.executionTraces]
        executionTraces = [trace for trace in executionTraces if trace is not None]

        presentRewards = DeepLearningAgent.computePresentRewards(executionTraces, config)

        return list(presentRewards)

    @staticmethod
    def createTrainingRewardNormalizer(execusionSessionIds, configDir):
        config = Configuration(configDir)

        rewardFutures = []

        rewardSequences = []

        cumulativeRewards = []

        longest = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=config['training_max_initialization_workers']) as executor:
            for sessionId in execusionSessionIds:
                rewardFutures.append(executor.submit(DeepLearningAgent.computeTotalRewardsParallel, str(sessionId), configDir))

            for rewardFuture in concurrent.futures.as_completed(rewardFutures):
                totalRewardSequence = rewardFuture.result()
                rewardSequences.append(totalRewardSequence)
                cumulativeRewards.append(numpy.sum(totalRewardSequence))
                longest = max(longest, len(totalRewardSequence))

        for totalRewardSequence in rewardSequences:
            while len(totalRewardSequence) < longest:
                # This isn't the greatest solution for handling execution sessions with different lengths, but it works
                # for now when we pretty much always have execution sessions of the same length except when debugging.
                totalRewardSequence.append(numpy.nan)

        rewardSequences = numpy.array(rewardSequences)
        trainingRewardNormalizer = sklearn.preprocessing.StandardScaler().fit(rewardSequences)

        cumulativeMean = numpy.mean(cumulativeRewards)
        cumulativeStd = numpy.std(cumulativeRewards)

        frameMean = numpy.mean(numpy.reshape(rewardSequences, newshape=[-1]))
        frameStd = numpy.std(numpy.reshape(rewardSequences, newshape=[-1]))

        print(datetime.now(), f"Reward baselines: Total reward per testing sequence: {cumulativeMean} +- std({cumulativeStd})")
        print(datetime.now(), f"                  Reward per frame: {frameMean} +- std({frameStd})")

        return trainingRewardNormalizer

    def augmentProcessedImageForTraining(self, processedImage):
        # Add random noise
        augmentedImage = processedImage + numpy.random.normal(loc=0, scale=self.config['training_image_gaussian_noise_scale'], size=processedImage.shape)

        # Clip the bounds to between 0 and 1
        augmentedImage = numpy.maximum(numpy.zeros_like(augmentedImage), numpy.minimum(numpy.ones_like(augmentedImage), augmentedImage))

        return augmentedImage


    def prepareBatchesForExecutionSession(self, executionSession, trainingRewardNormalizer=None):
        """
            This function prepares samples so that can be fed to the neural network. It will return a single
            batch containing all of the samples in this session. NOTE! It does not return a batch of the size
            set in the config files. This must be done by surrounding code. This will return a batch containing
            *all* samples in this execution session.

            :param executionSession:
            :param , trainingRewardNormalizer
            :return:
        """
        processedImages = []

        executionTraces = []

        videoPath = self.config.getKwolaUserDataDirectory("videos")
        for rawImage, traceId in zip(DeepLearningAgent.readVideoFrames(os.path.join(videoPath, f'{str(executionSession.id)}.mp4')), executionSession.executionTraces):
            trace = ExecutionTrace.loadFromDisk(traceId, self.config)
            if trace is not None:
                processedImage = DeepLearningAgent.processRawImageParallel(rawImage, self.config)
                processedImages.append(processedImage)
                executionTraces.append(trace)

        # First compute the present reward at each time step
        presentRewards = DeepLearningAgent.computePresentRewards(executionTraces, self.config)

        # If there is a normalizer, use it to normalize the rewards
        if trainingRewardNormalizer is not None:
            if len(presentRewards) < len(trainingRewardNormalizer.mean_):
               presentRewardsNormalizationInput = presentRewards + ([1] * (len(trainingRewardNormalizer.mean_) - len(presentRewards)))
            else:
                presentRewardsNormalizationInput = presentRewards

            # Compute the total rewards and normalize
            totalRewardsNormalizationInput = numpy.array([presentRewardsNormalizationInput])
            normalizedTotalRewards = trainingRewardNormalizer.transform(totalRewardsNormalizationInput)[0]

            presentRewards = normalizedTotalRewards[:len(presentRewards)]

        # Create the decaying future execution trace for the prediction algorithm
        tracesReversed = list(copy.copy(executionTraces))
        tracesReversed.reverse()
        currentTrace = numpy.zeros_like(executionTraces[0].branchExecutionTrace)
        executionTraceDiscountRate = self.config['future_execution_trace_decay_rate']
        decayingFutureBranchTraces = []
        for trace in tracesReversed:
            decayingFutureBranchTrace = numpy.array(trace.branchExecutionTrace)
            currentTrace *= executionTraceDiscountRate
            currentTrace += numpy.minimum(decayingFutureBranchTrace, numpy.ones_like(decayingFutureBranchTrace))
            decayingFutureBranchTraces.append(decayingFutureBranchTrace)

        decayingFutureBranchTraces.reverse()

        nextTraces = list(executionTraces)[1:]
        nextProcessedImages = list(processedImages)[1:]

        for trace, nextTrace, processedImage, nextProcessedImage, presentReward, decayingFutureBranchTrace in zip(executionTraces, nextTraces, processedImages, nextProcessedImages, presentRewards, decayingFutureBranchTraces):
            width = processedImage.shape[2]
            height = processedImage.shape[1]

            branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
            decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
            additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

            nextBranchFeature = numpy.minimum(nextTrace.startCumulativeBranchExecutionTrace, numpy.ones_like(nextTrace.startCumulativeBranchExecutionTrace))
            nextDecayingExecutionTraceFeature = numpy.array(nextTrace.startDecayingExecutionTrace)
            nextAdditionalFeature = numpy.concatenate([nextBranchFeature, nextDecayingExecutionTraceFeature], axis=0)

            pixelActionMap = self.createPixelActionMap(trace.actionMaps, height, width)
            nextPixelActionMap = self.createPixelActionMap(nextTrace.actionMaps, height, width)

            cursorVector = [0] * len(self.cursors)
            if trace.cursor in self.cursors:
                cursorVector[self.cursors.index(trace.cursor)] = 1
            else:
                cursorVector[self.cursors.index("none")] = 1

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

            rewardPixelMask = self.createRewardPixelMask(processedImage,
                                                         int(trace.actionPerformed.x * self.config['model_image_downscale_ratio']),
                                                         int(trace.actionPerformed.y * self.config['model_image_downscale_ratio'])
                                                         )

            # We down-sample some of the data points in the batch to be more compact.
            # We don't need a high precision for most of this data, so its better to be compact and save the ram
            # We also yield each sample individually instead of building up a list to save memory
            yield {
                "traceIds": [str(trace.id)],
                "processedImages": numpy.array([processedImage], dtype=numpy.float16),
                "additionalFeatures": numpy.array([additionalFeature], dtype=numpy.float16),
                "pixelActionMaps": numpy.array([pixelActionMap], dtype=numpy.uint8),
                "stepNumbers": numpy.array([trace.frameNumber - 1], dtype=numpy.int32),

                "nextProcessedImages": numpy.array([nextProcessedImage], dtype=numpy.float16),
                "nextAdditionalFeatures": numpy.array([nextAdditionalFeature], dtype=numpy.float16),
                "nextPixelActionMaps": numpy.array([nextPixelActionMap], dtype=numpy.uint8),
                "nextStepNumbers": numpy.array([nextTrace.frameNumber], dtype=numpy.uint8),

                "actionTypes": [trace.actionPerformed.type],
                "actionXs": numpy.array([int(trace.actionPerformed.x * self.config['model_image_downscale_ratio'])], dtype=numpy.int16),
                "actionYs": numpy.array([int(trace.actionPerformed.y * self.config['model_image_downscale_ratio'])], dtype=numpy.int16),
                "futureBranchTraces": numpy.array([decayingFutureBranchTrace], dtype=numpy.int8),

                "presentRewards": numpy.array([presentReward], dtype=numpy.float32),
                "rewardPixelMasks": numpy.array([rewardPixelMask], dtype=numpy.uint8),
                "executionFeatures": numpy.array([executionFeatures], dtype=numpy.uint8),
                "cursors": numpy.array([cursorVector], dtype=numpy.uint8)
            }

    def learnFromBatches(self, batches):
        """
            Runs backprop on the neural network with the given batch.

            :param batches: A list of batches of image/action/output pairs. Should be the return value from prepareBatchesForTestingStep
            :return:
        """
        totalLosses = []
        batchResultTensors = []
        self.optimizer.zero_grad()

        for batch in batches:
            rewardPixelMasks = self.variableWrapperFunc(torch.IntTensor, batch['rewardPixelMasks'])
            pixelActionMaps = self.variableWrapperFunc(torch.IntTensor, batch['pixelActionMaps'])
            nextStatePixelActionMaps = self.variableWrapperFunc(torch.IntTensor, batch['nextPixelActionMaps'])
            discountRate = self.variableWrapperFunc(torch.FloatTensor, [self.config['reward_discount_rate']])
            actionProbRewardSquareEdgeHalfSize = self.variableWrapperFunc(torch.IntTensor, [int(self.config['training_action_prob_reward_square_size']/2)])
            zeroTensor = self.variableWrapperFunc(torch.IntTensor, [0])
            oneTensor = self.variableWrapperFunc(torch.IntTensor, [1])
            oneTensorLong = self.variableWrapperFunc(torch.LongTensor, [1])
            oneTensorFloat = self.variableWrapperFunc(torch.FloatTensor, [1])
            stateValueLossWeightFloat = self.variableWrapperFunc(torch.FloatTensor, [self.config['loss_state_value_weight']])
            widthTensor = self.variableWrapperFunc(torch.IntTensor, [batch["processedImages"].shape[3]])
            heightTensor = self.variableWrapperFunc(torch.IntTensor, [batch["processedImages"].shape[2]])
            presentRewardsTensor = self.variableWrapperFunc(torch.FloatTensor, batch["presentRewards"])
            processedImagesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['processedImages']))
            additionalFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['additionalFeatures'])
            stepNumberTensor = self.variableWrapperFunc(torch.FloatTensor, batch['stepNumbers'])
            nextProcessedImagesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextProcessedImages']))
            nextAdditionalFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextAdditionalFeatures']))
            nextStepNumbers = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextStepNumbers']))

            if self.config['enable_trace_prediction_loss']:
                futureBranchTracesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['futureBranchTraces'])
            else:
                futureBranchTracesTensor = None

            if self.config['enable_execution_feature_prediction_loss']:
                executionFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['executionFeatures'])
            else:
                executionFeaturesTensor = None

            if self.config['enable_cursor_prediction_loss']:
                cursorsTensor = self.variableWrapperFunc(torch.FloatTensor, batch['cursors'])
            else:
                cursorsTensor = None

            currentStateOutputs = self.modelParallel({
                "image": processedImagesTensor,
                "additionalFeature": additionalFeaturesTensor,
                "pixelActionMaps": pixelActionMaps,
                "stepNumber": stepNumberTensor,
                "action_type": batch['actionTypes'],
                "action_x": batch['actionXs'],
                "action_y": batch['actionYs'],
                "outputStamp": False,
                "computeExtras": True,
                "computeActionProbabilities": True,
                "computeStateValues": True,
                "computeAdvantageValues": True,
                "computeRewards": True
            })

            presentRewardPredictions = currentStateOutputs['presentRewards']
            discountedFutureRewardPredictions = currentStateOutputs['discountFutureRewards']
            stateValuePredictions = currentStateOutputs['stateValues']
            advantagePredictions = currentStateOutputs['advantage']
            actionProbabilityPredictions = currentStateOutputs['actionProbabilities']

            with torch.no_grad():
                nextStateOutputs = self.targetNetwork({
                    "image": nextProcessedImagesTensor,
                    "additionalFeature": nextAdditionalFeaturesTensor,
                    "pixelActionMaps": nextStatePixelActionMaps,
                    "stepNumber": nextStepNumbers,
                    "outputStamp": False,
                    "computeExtras": False,
                    "computeActionProbabilities": False,
                    "computeStateValues": False,
                    "computeAdvantageValues": False,
                    "computeRewards": True
                })

                nextStatePresentRewardPredictions = nextStateOutputs['presentRewards']
                nextStateDiscountedFutureRewardPredictions = nextStateOutputs['discountFutureRewards']

            totalSampleLosses = []
            stateValueLosses = []
            advantageLosses = []
            actionProbabilityLosses = []
            presentRewardLosses = []
            targetHomogenizationLosses = []
            discountedFutureRewardLosses = []

            zippedValues = zip(range(len(presentRewardPredictions)), presentRewardPredictions, discountedFutureRewardPredictions,
                               nextStatePresentRewardPredictions, nextStateDiscountedFutureRewardPredictions,
                               rewardPixelMasks, presentRewardsTensor, stateValuePredictions, advantagePredictions,
                               batch['actionTypes'], batch['actionXs'], batch['actionYs'],
                               pixelActionMaps, actionProbabilityPredictions, batch['processedImages'])

            for sampleIndex, presentRewardImage, discountedFutureRewardImage,\
                nextStatePresentRewardImage, nextStateDiscountedFutureRewardImage,\
                rewardPixelMask, presentReward, stateValuePrediction, advantageImage,\
                actionType, actionX, actionY, pixelActionMap, actionProbabilityImage, processedImage in zippedValues:

                presentRewardsMasked = presentRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask
                discountedFutureRewardsMasked = discountedFutureRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask
                advantageMasked = advantageImage[self.actionsSorted.index(actionType)] * rewardPixelMask

                nextStatePresentRewardsMasked = nextStatePresentRewardImage
                nextStateDiscountedFutureRewardsMasked = nextStateDiscountedFutureRewardImage
                nextStateBestPossibleTotalReward = torch.max(nextStatePresentRewardsMasked + nextStateDiscountedFutureRewardsMasked)

                discountedFutureReward = nextStateBestPossibleTotalReward * discountRate

                torchBatchPresentRewards = torch.ones_like(presentRewardImage[self.actionsSorted.index(actionType)]) * presentReward * rewardPixelMask
                torchBatchDiscountedFutureRewards = torch.ones_like(discountedFutureRewardImage[self.actionsSorted.index(actionType)]) * discountedFutureReward * rewardPixelMask

                targetAdvantage = ((presentReward.detach() + discountedFutureReward.detach()) - stateValuePrediction.detach())
                targetAdvantageImage = torch.ones_like(advantageImage[self.actionsSorted.index(actionType)]) * targetAdvantage * rewardPixelMask

                bestActionX, bestActionY, bestActionType = self.getActionInfoTensorsFromRewardMap(advantageImage.detach())
                actionProbabilityTargetImage = torch.zeros_like(actionProbabilityImage)
                bestLeft = torch.max(bestActionX - actionProbRewardSquareEdgeHalfSize, zeroTensor)
                bestRight = torch.min(bestActionX + actionProbRewardSquareEdgeHalfSize, widthTensor-1)
                bestTop = torch.max(bestActionY - actionProbRewardSquareEdgeHalfSize, zeroTensor)
                bestBottom = torch.min(bestActionY + actionProbRewardSquareEdgeHalfSize, heightTensor-1)
                actionProbabilityTargetImage[bestActionType, bestTop:bestBottom, bestLeft:bestRight] = 1
                actionProbabilityTargetImage[bestActionType] *= pixelActionMap[bestActionType]
                countActionProbabilityTargetPixels = actionProbabilityTargetImage[bestActionType].sum()
                # The max here is just for safety, if any weird bugs happen we don't want any NaN values or division by zero
                actionProbabilityTargetImage[bestActionType] /= torch.max(oneTensorFloat, countActionProbabilityTargetPixels)

                # The max here is just for safety, if any weird bugs happen we don't want any NaN values or division by zero
                countPixelMask = torch.max(oneTensorLong, rewardPixelMask.sum())

                presentRewardLossMap = (torchBatchPresentRewards - presentRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
                discountedFutureRewardLossMap = (torchBatchDiscountedFutureRewards - discountedFutureRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
                advantageLossMap = (targetAdvantageImage - advantageMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
                actionProbabilityLossMap = (actionProbabilityTargetImage - actionProbabilityImage) * pixelActionMap

                presentRewardLoss = presentRewardLossMap.pow(2).sum() / countPixelMask
                discountedFutureRewardLoss = discountedFutureRewardLossMap.pow(2).sum() / countPixelMask
                advantageLoss = advantageLossMap.pow(2).sum() / countPixelMask
                actionProbabilityLoss = actionProbabilityLossMap.abs().sum()

                if self.config['enable_homogenization_loss']:
                    pixelFeatureImage = currentStateOutputs['pixelFeatureMap'][sampleIndex]

                    # Target Homogenization loss - basically, all of the pixels for the masked area should produce similar features to each other and different features
                    # from other pixels.
                    targetHomogenizationDifferenceMap = ((pixelFeatureImage - pixelFeatureImage[:, actionY, actionX].unsqueeze(1).unsqueeze(1)) * rewardPixelMask).pow(2).mean(axis=0)
                    targetDifferentiationDifferenceMap = ((pixelFeatureImage - pixelFeatureImage[:, actionY, actionX].unsqueeze(1).unsqueeze(1)) * (1.0 - rewardPixelMask)).pow(2).mean(axis=0)
                    targetHomogenizationLoss = torch.abs((targetHomogenizationDifferenceMap.sum() / countPixelMask) - (targetDifferentiationDifferenceMap.sum() / (widthTensor * heightTensor - countPixelMask)))
                    targetHomogenizationLosses.append(targetHomogenizationLoss.unsqueeze(0))

                presentRewardLosses.append(presentRewardLoss.unsqueeze(0))
                discountedFutureRewardLosses.append(discountedFutureRewardLoss.unsqueeze(0))
                advantageLosses.append(advantageLoss.unsqueeze(0))
                actionProbabilityLosses.append(actionProbabilityLoss.unsqueeze(0))

                stateValueLoss = (stateValuePrediction - (presentReward.detach() + discountedFutureReward.detach())).pow(2) * stateValueLossWeightFloat
                stateValueLosses.append(stateValueLoss)

                totalSampleLosses.append(presentRewardLoss + discountedFutureRewardLoss + advantageLoss + actionProbabilityLoss)

            extraLosses = []

            if self.config['enable_trace_prediction_loss']:
                tracePredictionLoss = (currentStateOutputs['predictedTraces'] - futureBranchTracesTensor).abs().mean()
                extraLosses.append(tracePredictionLoss.unsqueeze(0))
            else:
                tracePredictionLoss = zeroTensor


            if self.config['enable_execution_feature_prediction_loss']:
                predictedExecutionFeaturesLoss = (currentStateOutputs['predictedExecutionFeatures'] - executionFeaturesTensor).abs().mean()
                extraLosses.append(predictedExecutionFeaturesLoss.unsqueeze(0))
            else:
                predictedExecutionFeaturesLoss = zeroTensor


            if self.config['enable_cursor_prediction_loss']:
                predictedCursorLoss = (currentStateOutputs['predictedCursor'] - cursorsTensor).abs().mean()
                extraLosses.append(predictedCursorLoss.unsqueeze(0))
            else:
                predictedCursorLoss = zeroTensor

            presentRewardLoss = torch.mean(torch.cat(presentRewardLosses))
            discountedFutureRewardLoss = torch.mean(torch.cat(discountedFutureRewardLosses))
            stateValueLoss = torch.mean(torch.cat(stateValueLosses))
            advantageLoss = torch.mean(torch.cat(advantageLosses))
            actionProbabilityLoss = torch.mean(torch.cat(actionProbabilityLosses))
            totalRewardLoss = presentRewardLoss + discountedFutureRewardLoss + stateValueLoss + advantageLoss + actionProbabilityLoss

            if self.config['enable_homogenization_loss']:
                targetHomogenizationLoss = torch.mean(torch.cat(targetHomogenizationLosses))
                extraLosses.append(targetHomogenizationLoss.unsqueeze(0))
            else:
                targetHomogenizationLoss = zeroTensor

            if len(extraLosses) > 0:
                totalLoss = totalRewardLoss + torch.sum(torch.cat(extraLosses))
            else:
                totalLoss = totalRewardLoss

            totalRebalancedLoss = None
            if not self.config['enable_loss_balancing']:
                totalLoss.backward()
                totalLosses.append(totalLoss)
            else:
                if len(self.trainingLosses['totalLoss']) > 2:
                    averageStart = max(0, min(len(self.trainingLosses['totalRewardLoss']) - 1, self.config['loss_balancing_moving_average_period']))

                    runningAverageRewardLoss = numpy.mean(self.trainingLosses['totalRewardLoss'][-averageStart:])
                    runningAverageTracePredictionLoss = numpy.mean(self.trainingLosses['tracePredictionLoss'][-averageStart:])
                    runningAverageExecutionFeaturesLoss = numpy.mean(self.trainingLosses['predictedExecutionFeaturesLoss'][-averageStart:])
                    runningAverageHomogenizationLoss = numpy.mean(self.trainingLosses['targetHomogenizationLoss'][-averageStart:])
                    runningAveragePredictedCursorLoss = numpy.mean(self.trainingLosses['predictedCursorLoss'][-averageStart:])

                    tracePredictionAdustment = (runningAverageRewardLoss / (runningAverageTracePredictionLoss + 1e-6)) * self.config['loss_ratio_trace_prediction']
                    executionFeaturesAdustment = (runningAverageRewardLoss / (runningAverageExecutionFeaturesLoss + 1e-6)) * self.config['loss_ratio_execution_features']
                    homogenizationAdustment = (runningAverageRewardLoss / (runningAverageHomogenizationLoss + 1e-6)) * self.config['loss_ratio_homogenization']
                    predictedCursorAdustment = (runningAverageRewardLoss / (runningAveragePredictedCursorLoss + 1e-6)) * self.config['loss_ratio_predicted_cursor']
                else:
                    tracePredictionAdustment = 1
                    executionFeaturesAdustment = 1
                    homogenizationAdustment = 1
                    predictedCursorAdustment = 1

                if not self.config['enable_trace_prediction_loss']:
                    tracePredictionAdustment = 1

                if not self.config['enable_execution_feature_prediction_loss']:
                    executionFeaturesAdustment = 1

                if not self.config['enable_cursor_prediction_loss']:
                    predictedCursorAdustment = 1

                if not self.config['enable_homogenization_loss']:
                    homogenizationAdustment = 1

                totalRebalancedLoss = totalRewardLoss + \
                                      tracePredictionLoss * self.variableWrapperFunc(torch.FloatTensor, [tracePredictionAdustment]) + \
                                      predictedExecutionFeaturesLoss * self.variableWrapperFunc(torch.FloatTensor, [executionFeaturesAdustment]) + \
                                      targetHomogenizationLoss * self.variableWrapperFunc(torch.FloatTensor, [homogenizationAdustment]) + \
                                      predictedCursorLoss * self.variableWrapperFunc(torch.FloatTensor, [predictedCursorAdustment])

                totalRebalancedLoss.backward()
                totalLosses.append(totalRebalancedLoss)

            batchResultTensors.append((
                presentRewardLoss,
                discountedFutureRewardLoss,
                stateValueLoss,
                advantageLoss,
                actionProbabilityLoss,
                tracePredictionLoss,
                predictedExecutionFeaturesLoss,
                targetHomogenizationLoss,
                predictedCursorLoss,
                totalRewardLoss,
                totalLoss,
                totalRebalancedLoss,
                totalSampleLosses,
                batch
            ))


        # Put a check in so that we don't do the optimizer step if there are NaNs in the loss
        if numpy.count_nonzero(numpy.isnan([totalLoss.data.item() for totalLoss in totalLosses])) == 0:
            self.optimizer.step()
        else:
            print(datetime.now(), "ERROR! NaN detected in loss calculation. Skipping optimization step.")
            for batchIndex, batchResult in batchResultTensors:
                presentRewardLoss, discountedFutureRewardLoss, stateValueLoss, \
                advantageLoss,actionProbabilityLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, \
                targetHomogenizationLoss, predictedCursorLoss, totalRewardLoss, totalLoss, totalRebalancedLoss, \
                totalSampleLosses, batch = batchResult

                print(datetime.now(), "Batch", batchIndex)
                print(datetime.now(), "presentRewardLoss", float(presentRewardLoss.data.item()))
                print(datetime.now(), "discountedFutureRewardLoss", float(discountedFutureRewardLoss.data.item()))
                print(datetime.now(), "stateValueLoss", float(stateValueLoss.data.item()))
                print(datetime.now(), "advantageLoss", float(advantageLoss.data.item()))
                print(datetime.now(), "actionProbabilityLoss", float(actionProbabilityLoss.data.item()))
                print(datetime.now(), "tracePredictionLoss", float(tracePredictionLoss.data.item()))
                print(datetime.now(), "predictedExecutionFeaturesLoss", float(predictedExecutionFeaturesLoss.data.item()))
                print(datetime.now(), "targetHomogenizationLoss", float(targetHomogenizationLoss.data.item()))
                print(datetime.now(), "predictedCursorLoss", float(predictedCursorLoss.data.item()), flush=True)

            return

        batchResults = []

        for batchResult in batchResultTensors:
            presentRewardLoss, discountedFutureRewardLoss, stateValueLoss, \
            advantageLoss,actionProbabilityLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, \
            targetHomogenizationLoss, predictedCursorLoss, totalRewardLoss, totalLoss, totalRebalancedLoss, \
            totalSampleLosses, batch = batchResult

            totalRewardLoss = float(totalRewardLoss.data.item())
            presentRewardLoss = float(presentRewardLoss.data.item())
            discountedFutureRewardLoss = float(discountedFutureRewardLoss.data.item())
            stateValueLoss = float(stateValueLoss.data.item())
            advantageLoss = float(advantageLoss.data.item())
            actionProbabilityLoss = float(actionProbabilityLoss.data.item())
            tracePredictionLoss = float(tracePredictionLoss.data.item())
            predictedExecutionFeaturesLoss = float(predictedExecutionFeaturesLoss.data.item())
            targetHomogenizationLoss = float(targetHomogenizationLoss.data.item())
            predictedCursorLoss = float(predictedCursorLoss.data.item())
            totalLoss = float(totalLoss.data.item())
            if self.config['enable_loss_balancing']:
                totalRebalancedLoss = float(totalRebalancedLoss.data.item())
            else:
                totalRebalancedLoss = 0

            batchReward = float(numpy.sum(batch['presentRewards']))

            self.trainingLosses["totalRewardLoss"].append(totalRewardLoss)
            self.trainingLosses["presentRewardLoss"].append(presentRewardLoss)
            self.trainingLosses["discountedFutureRewardLoss"].append(discountedFutureRewardLoss)
            self.trainingLosses["stateValueLoss"].append(stateValueLoss)
            self.trainingLosses["advantageLoss"].append(advantageLoss)
            self.trainingLosses["actionProbabilityLoss"].append(actionProbabilityLoss)
            self.trainingLosses["tracePredictionLoss"].append(tracePredictionLoss)
            self.trainingLosses["predictedExecutionFeaturesLoss"].append(predictedExecutionFeaturesLoss)
            self.trainingLosses["targetHomogenizationLoss"].append(targetHomogenizationLoss)
            self.trainingLosses["predictedCursorLoss"].append(predictedCursorLoss)
            self.trainingLosses["totalLoss"].append(totalLoss)
            self.trainingLosses["totalRebalancedLoss"].append(totalRebalancedLoss)

            sampleLosses = [tensor.data.item() for tensor in totalSampleLosses]

            batchResults.append((totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, stateValueLoss, advantageLoss, actionProbabilityLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, targetHomogenizationLoss, predictedCursorLoss, totalLoss, totalRebalancedLoss, batchReward, sampleLosses))
        return batchResults

    @staticmethod
    def processRawImageParallel(rawImage, config):
        width = rawImage.shape[1]
        height = rawImage.shape[0]

        grey = skimage.color.rgb2gray(rawImage[:, :, :3])

        shrunkWidth = int(width * config['model_image_downscale_ratio'])
        shrunkHeight = int(height * config['model_image_downscale_ratio'])

        # Make sure the image aligns to the nearest 8 pixels,
        # this is because the image gets downsampled and upsampled
        # within the neural network by 8x, so both the image width
        # and image height must perfectly divide 8 or else there
        # will be errors within the neural network.
        if (shrunkWidth % 8) > 0:
            shrunkWidth += 8 - (shrunkWidth % 8)

        if (shrunkHeight % 8) > 0:
            shrunkHeight += 8 - (shrunkHeight % 8)

        shrunk = skimage.transform.resize(grey, (shrunkHeight, shrunkWidth), anti_aliasing=True)

        # Convert to grey-scale image
        processedImage = numpy.array([shrunk])

        # Round the float values down to 0. This minimizes some the error introduced by the video codecs
        processedImage = numpy.around(processedImage, decimals=2)

        return processedImage
