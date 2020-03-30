from .BaseAgent import BaseAgent
from .BradNet import BradNet
from kwola.components.utilities.debug_plot import showRewardImageDebug
from kwola.config import config
from kwola.models.ExecutionTraceModel import ExecutionTrace
from kwola.models.ExecutionSessionModel import ExecutionSession
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.actions.ClickTapAction import ClickTapAction
from kwola.models.actions.RightClickAction import RightClickAction
from kwola.models.actions.TypeAction import TypeAction
from kwola.models.actions.WaitAction import WaitAction
from skimage.segmentation import felzenszwalb, mark_boundaries
import bz2
import concurrent.futures
import cv2
import bson
import cv2
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

    def __init__(self, agentConfiguration, whichGpu="all"):
        super().__init__()

        self.agentConfiguration = agentConfiguration
        
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
                self.model.load_state_dict(torch.load(self.modelPath, map_location=device))
            elif self.whichGpu != "all":
                device = torch.device(f"cuda:{self.whichGpu}")
                self.model.load_state_dict(torch.load(self.modelPath, map_location=device))
            else:
                self.model.load_state_dict(torch.load(self.modelPath))

            if self.whichGpu is None:
                device = torch.device('cpu')
                self.targetNetwork.load_state_dict(torch.load(self.modelPath, map_location=device))
            elif self.whichGpu != "all":
                device = torch.device(f"cuda:{self.whichGpu}")
                self.targetNetwork.load_state_dict(torch.load(self.modelPath, map_location=device))
            else:
                self.targetNetwork.load_state_dict(torch.load(self.modelPath))


    def save(self):
        """
            Saves the agent to the db / disk.

            :return:
        """
        torch.save(self.model.state_dict(), self.modelPath)


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

        self.model = BradNet(self.agentConfiguration, branchFeatureSize * 2, len(self.actions), branchFeatureSize, 12, len(self.cursors))
        self.targetNetwork = BradNet(self.agentConfiguration, branchFeatureSize * 2, len(self.actions), branchFeatureSize, 12, len(self.cursors))

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
                                      lr=self.agentConfiguration['training_learning_rate'],
                                      betas=(self.agentConfiguration['training_gradient_exponential_moving_average_decay'],
                                             self.agentConfiguration['training_gradient_squared_exponential_moving_average_decay'])
                                      )

    def updateTargetNetwork(self):
        self.targetNetwork.load_state_dict(self.model.state_dict())

    def processImages(self, images):
        convertedImageFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for image in images:
                convertedImageFuture = executor.submit(processRawImageParallel, image)
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
                pixelActionMap[actionTypeIndex, int(element['top']):int(element['bottom']), int(element['left']):int(element['right'])] = 1

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

        for sampleIndex, image, additionalFeatureVector, sampleActionMaps, sampleRecentActions in zip(range(len(processedImages)), processedImages, additionalFeatures, envActionMaps, recentActions):
            randomActionProbability = (float(sampleIndex + 1) / float(len(processedImages))) * 0.50 * (1 + (stepNumber / self.agentConfiguration['testing_sequence_length']))
            weightedRandomActionProbability = (float(sampleIndex + 1) / float(len(processedImages))) * 0.50 * (1 + (stepNumber / self.agentConfiguration['testing_sequence_length']))

            filteredSampleActionMaps = []
            filteredSampleActionRecentActionCounts = []
            for map in sampleActionMaps:
                # Check to see if the map is out of the screen
                if map.top > height or map.bottom < 0 or map.left > width or map.right < 0:
                    # skip this action map, don't add it to the filtered list
                    continue

                count = 0
                for recentAction in sampleRecentActions:
                    for recentActionMap in self.getActionMapsIntersectingWithAction(recentAction, recentAction.actionMapsAvailable):
                        if map == recentActionMap:
                            count += 1
                            break
                if count < self.agentConfiguration['testing_max_repeat_action_maps_without_new_branches']:
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
            else:
                actionMapWeights = numpy.array([self.elementBaseWeights.get(map.elementType, 0.5) for map in sampleActionMaps]) / (numpy.array(sampleActionRecentActionCounts) + 1)

                chosenActionMapIndex = numpy.random.choice(range(len(sampleActionMaps)), p=scipy.special.softmax(actionMapWeights))
                chosenActionMap = sampleActionMaps[chosenActionMapIndex]

                actionX = random.randint(max(0, min(width - 1, chosenActionMap.left)), max(0, min(chosenActionMap.right - 1, width - 1)))
                actionY = random.randint(max(0, min(height - 1, chosenActionMap.top)), max(0, min(chosenActionMap.bottom - 1, height - 1)))

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

                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = "random"
                action.predictedReward = None
                action.wasRepeatOverride = False
                action.actionMapsAvailable = sampleActionMaps
                actions.append((sampleIndex, action))

        if len(imageBatch) > 0:
            imageTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(imageBatch))
            additionalFeatureTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(additionalFeatureVectorBatch))
            pixelActionMapTensor = self.variableWrapperFunc(torch.FloatTensor, pixelActionMapsBatch)

            with torch.no_grad():
                self.model.eval()

                outputs = self.modelParallel({
                    "image": imageTensor,
                    "additionalFeature": additionalFeatureTensor,
                    "pixelActionMaps": pixelActionMapTensor,
                    "outputStamp": False,
                    "computeExtras": False,
                    "computeRewards": False,
                    "computeActionProbabilities": True,
                    "computeStateValues": False,
                    "computeAdvantageValues": True
                })

                actionProbabilities = outputs['actionProbabilities'].cpu()
                # actionProbabilities = outputs['advantage'].cpu()

            for sampleIndex, sampleEpsilon, sampleActionProbs, sampleRecentActions, sampleActionMaps in zip(batchSampleIndexes, epsilonsPerSample, actionProbabilities, recentActionsBatch, actionMapsBatch):
                weighted = bool(random.random() < sampleEpsilon)
                override = False
                source = None
                actionType = None
                actionX = None
                actionY = None
                samplePredictedReward = None
                if not weighted:
                    source = "prediction"

                    actionX, actionY, actionType  = self.getActionInfoTensorsFromRewardMap(sampleActionProbs)
                    actionX = actionX.data.item()
                    actionY = actionY.data.item()
                    actionType = actionType.data.item()
                    
                    samplePredictedReward = sampleActionProbs[actionType, actionY, actionX].data.item()

                    potentialAction = self.actions[self.actionsSorted[actionType]](actionX, actionY)
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
                                if recentMap == potentialMap:
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
                    try:
                        actionIndex = numpy.random.choice(range(len(self.actionsSorted) * height * width), p=reshaped)
                    except ValueError:
                        print(datetime.now(), "Error in weighted random choice! Probabilities do not all add up to 1.", flush=True)
                        # This usually occurs when all the probabilities do not add up to 1, due to floating point error.
                        # So instead we just pick an action randomly with no probabilities.
                        actionIndex = numpy.random.choice(range(len(self.actionsSorted) * height * width))

                    newProbs = numpy.zeros([len(self.actionsSorted) * height * width])
                    newProbs[actionIndex] = 1

                    newProbs = newProbs.reshape([len(self.actionsSorted), height * width])
                    actionType = newProbs.max(axis=1).argmax(axis=0)
                    newProbs = newProbs.reshape([len(self.actionsSorted), height, width])
                    actionY = newProbs[actionType].max(axis=1).argmax(axis=0)
                    actionX = newProbs[actionType, actionY].argmax(axis=0)

                    source = "weighted_random"
                    samplePredictedReward = sampleActionProbs[actionType, actionY, actionX].data.item()

                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = source
                action.predictedReward = samplePredictedReward
                action.actionMapsAvailable = sampleActionMaps
                action.wasRepeatOverride = override
                actions.append((sampleIndex, action))

        sortedActions = sorted(actions, key=lambda row: row[0])

        return [action[1] for action in sortedActions]

    @staticmethod
    def computePresentRewards(executionSession):
        agentConfiguration = config.getAgentConfiguration()

        # First compute the present reward at each time step
        presentRewards = []
        for trace in executionSession.executionTraces:
            tracePresentReward = 0.0

            if trace.didActionSucceed:
                tracePresentReward += agentConfiguration['reward_action_success']
            else:
                tracePresentReward += agentConfiguration['reward_action_failure']

            if trace.didCodeExecute:
                tracePresentReward += agentConfiguration['reward_code_executed']
            else:
                tracePresentReward += agentConfiguration['reward_no_code_executed']

            if trace.didNewBranchesExecute:
                tracePresentReward += agentConfiguration['reward_new_code_executed']
            else:
                tracePresentReward += agentConfiguration['reward_no_new_code_executed']

            if trace.hadNetworkTraffic:
                tracePresentReward += agentConfiguration['reward_network_traffic']
            else:
                tracePresentReward += agentConfiguration['reward_no_network_traffic']

            if trace.hadNewNetworkTraffic:
                tracePresentReward += agentConfiguration['reward_new_network_traffic']
            else:
                tracePresentReward += agentConfiguration['reward_no_new_network_traffic']

            if trace.didScreenshotChange:
                tracePresentReward += agentConfiguration['reward_screenshot_changed']
            else:
                tracePresentReward += agentConfiguration['reward_no_screenshot_change']

            if trace.isScreenshotNew:
                tracePresentReward += agentConfiguration['reward_new_screenshot']
            else:
                tracePresentReward += agentConfiguration['reward_no_new_screenshot']

            if trace.didURLChange:
                tracePresentReward += agentConfiguration['reward_url_changed']
            else:
                tracePresentReward += agentConfiguration['reward_no_url_change']

            if trace.isURLNew:
                tracePresentReward += agentConfiguration['reward_new_url']
            else:
                tracePresentReward += agentConfiguration['reward_no_new_url']

            if trace.hadLogOutput:
                tracePresentReward += agentConfiguration['reward_log_output']
            else:
                tracePresentReward += agentConfiguration['reward_no_log_output']

            presentRewards.append(tracePresentReward)
        return presentRewards

    @staticmethod
    def computeDiscountedFutureRewards(executionSession):
        agentConfiguration = config.getAgentConfiguration()

        # First compute the present reward at each time step
        presentRewards = DeepLearningAgent.computePresentRewards(executionSession)

        # Now compute the discounted reward
        discountedFutureRewards = []
        presentRewards.reverse()
        current = 0
        for reward in presentRewards:
            current *= agentConfiguration['reward_discount_rate']
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
        videoPath = config.getKwolaUserDataDirectory("videos")

        rawImages = DeepLearningAgent.readVideoFrames(os.path.join(videoPath, f"{str(executionSession.id)}.mp4"))

        presentRewards = DeepLearningAgent.computePresentRewards(executionSession)

        discountedFutureRewards = DeepLearningAgent.computeDiscountedFutureRewards(executionSession)

        tempScreenshotDirectory = tempfile.mkdtemp()

        debugImageIndex = 0

        lastRawImage = rawImages.pop(0)

        mpl.use('Agg')
        mpl.rcParams['figure.max_open_warning'] = 1000

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for trace, rawImage in zip(executionSession.executionTraces, rawImages):
                future = executor.submit(self.createDebugImagesForExecutionTrace, str(executionSession.id), debugImageIndex, trace.to_json(), rawImage, lastRawImage, presentRewards, discountedFutureRewards, tempScreenshotDirectory)
                futures.append(future)

                # self.createDebugImagesForExecutionTrace(str(executionSession.id), debugImageIndex, trace, rawImage, lastRawImage, presentRewards, discountedFutureRewards, tempScreenshotDirectory)

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
                cropLeft, cropTop, cropRight, cropBottom = self.calculateTrainingCropPosition(trace, imageWidth, imageHeight)

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

                processedImage = processRawImageParallel(rawImage)

                rewardPixelMaskAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                currentFig += 1
                rewardPixelMask = self.createRewardPixelMask(processedImage, trace.actionPerformed.x, trace.actionPerformed.y)
                rewardPixelCount = numpy.count_nonzero(rewardPixelMask)
                rewardPixelMaskAxes.imshow(rewardPixelMask, vmin=0, vmax=1, cmap=plt.get_cmap("gray"), interpolation="bilinear")
                rewardPixelMaskAxes.set_xticks([])
                rewardPixelMaskAxes.set_yticks([])
                rewardPixelMaskAxes.set_title(f"{rewardPixelCount} target pixels")

                # pixelActionMapAxes = mainFigure.add_subplot(numColumns, numRows, currentFig)
                # currentFig += 1
                pixelActionMap = self.createPixelActionMap(trace.actionPerformed.actionMapsAvailable, imageHeight, imageWidth)
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
                    minValue = numpy.min(numpy.array(totalRewardPredictions[0][actionIndex]))

                    rewardPredictionAxes[actionIndex].set_xticks([])
                    rewardPredictionAxes[actionIndex].set_yticks([])

                    rewardPredictionsShrunk = skimage.measure.block_reduce(totalRewardPredictions[0][actionIndex], (3,3), numpy.max)

                    im = rewardPredictionAxes[actionIndex].imshow(rewardPredictionsShrunk, cmap=mainColorMap, interpolation="nearest")
                    rewardPredictionAxes[actionIndex].set_title(f"{action} {minValue:.2f}-{maxValue:.2f}")
                    mainFigure.colorbar(im, ax=rewardPredictionAxes[actionIndex], orientation='vertical')

                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(advantagePredictions[0][actionIndex]))
                    minValue = numpy.min(numpy.array(advantagePredictions[0][actionIndex]))

                    advantagePredictionAxes[actionIndex].set_xticks([])
                    advantagePredictionAxes[actionIndex].set_yticks([])

                    advanagePredictionsShrunk = skimage.measure.block_reduce(advantagePredictions[0][actionIndex], (3, 3), numpy.max)

                    im = advantagePredictionAxes[actionIndex].imshow(advanagePredictionsShrunk, cmap=mainColorMap, interpolation="nearest")
                    advantagePredictionAxes[actionIndex].set_title(f"{action} {minValue:.2f}-{maxValue:.2f}")
                    mainFigure.colorbar(im, ax=advantagePredictionAxes[actionIndex], orientation='vertical')

                for actionIndex, action in enumerate(self.actionsSorted):
                    maxValue = numpy.max(numpy.array(actionProbabilities[0][actionIndex]))
                    minValue = numpy.min(numpy.array(actionProbabilities[0][actionIndex]))

                    actionProbabilityPredictionAxes[actionIndex].set_xticks([])
                    actionProbabilityPredictionAxes[actionIndex].set_yticks([])

                    actionProbabilityPredictionsShrunk = skimage.measure.block_reduce(actionProbabilities[0][actionIndex], (3, 3), numpy.max)

                    im = actionProbabilityPredictionAxes[actionIndex].imshow(actionProbabilityPredictionsShrunk, cmap=mainColorMap, interpolation="nearest")
                    actionProbabilityPredictionAxes[actionIndex].set_title(f"{action} {minValue:.3f}-{maxValue:.3f}")
                    mainFigure.colorbar(im, ax=actionProbabilityPredictionAxes[actionIndex], orientation='vertical')

                stampAxes.set_xticks([])
                stampAxes.set_yticks([])
                stampIm = stampAxes.imshow(stamp.data[0], cmap=greyColorMap, interpolation="nearest", vmin=-20.0, vmax=20.0)
                mainFigure.colorbar(stampIm, ax=stampAxes, orientation='vertical')
                stampAxes.set_title("Memory Stamp")

                stateValueAxes.set_xticks([])
                stateValueAxes.set_yticks([])
                stateValueIm = stateValueAxes.imshow([[stateValue]], cmap=mainColorMap, interpolation="nearest", vmin=-2.0, vmax=15.0)
                mainFigure.colorbar(stateValueIm, ax=stateValueAxes, orientation='vertical')
                stateValueAxes.set_title(f"State Value {stateValue:.3f}")

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


    def calculateTrainingCropPosition(self, trace, imageWidth, imageHeight, nextStepCrop=False):
        cropWidth = self.agentConfiguration['training_crop_width']
        cropHeight = self.agentConfiguration['training_crop_height']
        if nextStepCrop:
            cropWidth = self.agentConfiguration['training_next_step_crop_width']
            cropHeight = self.agentConfiguration['training_next_step_crop_height']

        cropLeft = trace.actionPerformed.x - cropWidth / 2
        cropTop = trace.actionPerformed.y - cropHeight / 2

        cropLeft = max(0, cropLeft)
        cropLeft = min(imageWidth - cropWidth, cropLeft)

        cropTop = max(0, cropTop)
        cropTop = min(imageHeight - cropHeight, cropTop)

        cropRight = cropLeft + cropWidth
        cropBottom = cropTop + cropHeight

        return (int(cropLeft), int(cropTop), int(cropRight), int(cropBottom))


    @staticmethod
    def computeTotalRewardsParallel(execusionSessionId):
        session = ExecutionSession.objects(id=bson.ObjectId(execusionSessionId)).no_dereference().first()

        session.executionTraces = [
            ExecutionTrace.objects(id=traceId['_ref'].id).exclude("branchExecutionTraceCompressed", "startDecayingExecutionTraceCompressed", "startCumulativeBranchExecutionTraceCompressed").first()
            for traceId in session.executionTraces
        ]

        presentRewards = DeepLearningAgent.computePresentRewards(session)
        # futureRewards = DeepLearningAgent.computeDiscountedFutureRewards(session)

        # return [present + future for present, future in zip(presentRewards, futureRewards)]
        return list(presentRewards)

    @staticmethod
    def createTrainingRewardNormalizer(execusionSessionIds):
        agentConfig = config.getAgentConfiguration()

        rewardFutures = []

        rewardSequences = []

        cumulativeRewards = []

        longest = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=agentConfig['training_max_initialization_workers']) as executor:
            for sessionId in execusionSessionIds:
                rewardFutures.append(executor.submit(DeepLearningAgent.computeTotalRewardsParallel, str(sessionId)))

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
        augmentedImage = processedImage + numpy.random.normal(loc=0, scale=self.agentConfiguration['training_image_gaussian_noise_scale'], size=processedImage.shape)

        # Clip the bounds to between 0 and 1
        augmentedImage = numpy.maximum(numpy.zeros_like(augmentedImage), numpy.minimum(numpy.ones_like(augmentedImage), augmentedImage))

        return augmentedImage


    def prepareBatchForExecutionSession(self, executionSession, trainingRewardNormalizer=None):
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

        videoPath = config.getKwolaUserDataDirectory("videos")
        for rawImage in DeepLearningAgent.readVideoFrames(os.path.join(videoPath, f'{str(executionSession.id)}.mp4')):
            processedImage = processRawImageParallel(rawImage)
            processedImages.append(processedImage)

        # First compute the present reward at each time step
        presentRewards = DeepLearningAgent.computePresentRewards(executionSession)

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

        batchTraceIds = []
        batchProcessedImages = []
        batchAdditionalFeatures = []
        batchPixelActionMaps = []

        batchNextStateProcessedImages = []
        batchNextStateAdditionalFeatures = []
        batchNextStatePixelActionMaps = []

        batchActionTypes = []
        batchActionXs = []
        batchActionYs = []
        batchExecutionTraces = []
        # batchDiscountedFutureRewards = []
        batchPresentRewards = []
        batchRewardPixelMasks = []
        batchExecutionFeatures = []
        batchCursors = []

        nextTraces = list(executionSession.executionTraces)[1:]
        nextProcessedImages = list(processedImages)[1:]

        for trace, nextTrace, processedImage, nextProcessedImage, presentReward, executionTrace in zip(executionSession.executionTraces, nextTraces, processedImages, nextProcessedImages, presentRewards, executionTraces):
            width = processedImage.shape[2]
            height = processedImage.shape[1]

            batchTraceIds.append(str(trace.id))

            cropLeft, cropTop, cropRight, cropBottom = self.calculateTrainingCropPosition(trace, imageWidth=width, imageHeight=height)

            branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
            decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
            additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

            nextBranchFeature = numpy.minimum(nextTrace.startCumulativeBranchExecutionTrace, numpy.ones_like(nextTrace.startCumulativeBranchExecutionTrace))
            nextDecayingExecutionTraceFeature = numpy.array(nextTrace.startDecayingExecutionTrace)
            nextAdditionalFeature = numpy.concatenate([nextBranchFeature, nextDecayingExecutionTraceFeature], axis=0)

            batchProcessedImages.append(processedImage[:, cropTop:cropBottom, cropLeft:cropRight])
            batchAdditionalFeatures.append(additionalFeature)

            batchNextStateProcessedImages.append(nextProcessedImage) # We don't crop the next state, otherwise we couldn't get an accurate estimate of the best possible future value
            batchNextStateAdditionalFeatures.append(nextAdditionalFeature)

            pixelActionMap = self.createPixelActionMap(trace.actionMaps, height, width)
            batchPixelActionMaps.append(pixelActionMap[:, cropTop:cropBottom, cropLeft:cropRight])

            nextPixelActionMap = self.createPixelActionMap(nextTrace.actionMaps, height, width)
            batchNextStatePixelActionMaps.append(nextPixelActionMap)

            batchActionTypes.append(trace.actionPerformed.type)
            batchActionXs.append(trace.actionPerformed.x - cropLeft)
            batchActionYs.append(trace.actionPerformed.y - cropTop)

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

            # batchDiscountedFutureRewards.append(discountedFutureReward)
            batchPresentRewards.append(presentReward)

            rewardPixelMask = self.createRewardPixelMask(processedImage, trace.actionPerformed.x, trace.actionPerformed.y)

            batchRewardPixelMasks.append(rewardPixelMask[cropTop:cropBottom, cropLeft:cropRight])

            batchExecutionFeatures.append(executionFeatures)

        if len(batchProcessedImages) == 0:
            print("Error while preparing the batch. No resulting processed images.", flush=True)

        # We down-sample some of the data points in the batch to be more compact.
        # We don't need a high precision for most of this data, so its better to be compact and save the ram
        return {
            "traceIds": batchTraceIds,
            "processedImages": numpy.array(batchProcessedImages, dtype=numpy.float16),
            "additionalFeatures": numpy.array(batchAdditionalFeatures, dtype=numpy.float16),
            "pixelActionMaps": numpy.array(batchPixelActionMaps, dtype=numpy.uint8),

            "nextProcessedImages": numpy.array(batchNextStateProcessedImages, dtype=numpy.float16),
            "nextAdditionalFeatures": numpy.array(batchNextStateAdditionalFeatures, dtype=numpy.float16),
            "nextPixelActionMaps": numpy.array(batchNextStatePixelActionMaps, dtype=numpy.uint8),

            "actionTypes": batchActionTypes,
            "actionXs": numpy.array(batchActionXs, dtype=numpy.int16),
            "actionYs": numpy.array(batchActionYs, dtype=numpy.int16),
            "executionTraces": numpy.array(batchExecutionTraces, dtype=numpy.int8),
            # "discountedFutureRewards": numpy.array(batchDiscountedFutureRewards, dtype=numpy.float32),
            "presentRewards": numpy.array(batchPresentRewards, dtype=numpy.float32),
            "rewardPixelMasks": numpy.array(batchRewardPixelMasks, dtype=numpy.uint8),
            "executionFeatures": numpy.array(batchExecutionFeatures, dtype=numpy.uint8),
            "cursors": numpy.array(batchCursors, dtype=numpy.uint8)
        }

    def learnFromBatch(self, batch):
        """
            Runs backprop on the neural network with the given batch.

            :param batch: A batch of image/action/output pairs. Should be the return value from prepareBatchesForTestingSequence
            :return:
        """
        rewardPixelMasks = self.variableWrapperFunc(torch.IntTensor, batch['rewardPixelMasks'])
        pixelActionMaps = self.variableWrapperFunc(torch.IntTensor, batch['pixelActionMaps'])
        nextStatePixelActionMaps = self.variableWrapperFunc(torch.IntTensor, batch['nextPixelActionMaps'])
        discountRate = self.variableWrapperFunc(torch.FloatTensor, [self.agentConfiguration['reward_discount_rate']])
        actionProbRewardSquareEdgeHalfSize = self.variableWrapperFunc(torch.IntTensor, [int(self.agentConfiguration['training_action_prob_reward_square_size']/2)])
        zeroTensor = self.variableWrapperFunc(torch.IntTensor, [0])
        widthTensor = self.variableWrapperFunc(torch.IntTensor, [batch["processedImages"].shape[3]])
        heightTensor = self.variableWrapperFunc(torch.IntTensor, [batch["processedImages"].shape[2]])
        presentRewardsTensor = self.variableWrapperFunc(torch.FloatTensor, batch["presentRewards"])
        processedImagesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['processedImages']))
        additionalFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['additionalFeatures'])
        nextProcessedImagesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextProcessedImages']))
        nextAdditionalFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, numpy.array(batch['nextAdditionalFeatures']))

        if self.agentConfiguration['enable_trace_prediction_loss']:
            executionTracesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['executionTraces'])
        else:
            executionTracesTensor = None

        if self.agentConfiguration['enable_execution_feature_prediction_loss']:
            executionFeaturesTensor = self.variableWrapperFunc(torch.FloatTensor, batch['executionFeatures'])
        else:
            executionFeaturesTensor = None

        if self.agentConfiguration['enable_cursor_prediction_loss']:
            cursorsTensor = self.variableWrapperFunc(torch.FloatTensor, batch['cursors'])
        else:
            cursorsTensor = None

        currentStateOutputs = self.modelParallel({
            "image": processedImagesTensor,
            "additionalFeature": additionalFeaturesTensor,
            "pixelActionMaps": pixelActionMaps,
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
                "outputStamp": False,
                "computeExtras": False,
                "computeActionProbabilities": False,
                "computeStateValues": False,
                "computeAdvantageValues": False,
                "computeRewards": True
            })

            nextStatePresentRewardPredictions = nextStateOutputs['presentRewards']
            nextStateDiscountedFutureRewardPredictions = nextStateOutputs['discountFutureRewards']

        totalRewardLosses = []
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
            actionProbabilityTargetImage[bestActionType] /= countActionProbabilityTargetPixels

            countPixelMask = (rewardPixelMask.sum())

            presentRewardLossMap = (torchBatchPresentRewards - presentRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
            discountedFutureRewardLossMap = (torchBatchDiscountedFutureRewards - discountedFutureRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
            advantageLossMap = (targetAdvantageImage - advantageMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
            actionProbabilityLossMap = (actionProbabilityTargetImage - actionProbabilityImage) * pixelActionMap

            presentRewardLoss = presentRewardLossMap.pow(2).sum() / countPixelMask
            discountedFutureRewardLoss = discountedFutureRewardLossMap.pow(2).sum() / countPixelMask
            advantageLoss = advantageLossMap.pow(2).sum() / countPixelMask
            actionProbabilityLoss = actionProbabilityLossMap.abs().sum()

            if self.agentConfiguration['enable_homogenization_loss']:
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

            stateValueLoss = (stateValuePrediction - (presentReward.detach() + discountedFutureReward.detach())).pow(2)
            stateValueLosses.append(stateValueLoss)

            totalRewardLosses.append(presentRewardLoss + discountedFutureRewardLoss + advantageLoss + actionProbabilityLoss)

        extraLosses = []

        if self.agentConfiguration['enable_trace_prediction_loss']:
            tracePredictionLoss = (currentStateOutputs['predictedTraces'] - executionTracesTensor).abs().mean()
            extraLosses.append(tracePredictionLoss)
        else:
            tracePredictionLoss = zeroTensor


        if self.agentConfiguration['enable_execution_feature_prediction_loss']:
            predictedExecutionFeaturesLoss = (currentStateOutputs['predictedExecutionFeatures'] - executionFeaturesTensor).abs().mean()
            extraLosses.append(predictedExecutionFeaturesLoss)
        else:
            predictedExecutionFeaturesLoss = zeroTensor


        if self.agentConfiguration['enable_cursor_prediction_loss']:
            predictedCursorLoss = (currentStateOutputs['predictedCursor'] - cursorsTensor).abs().mean()
            extraLosses.append(predictedCursorLoss)
        else:
            predictedCursorLoss = zeroTensor

        presentRewardLoss = torch.mean(torch.cat(presentRewardLosses))
        discountedFutureRewardLoss = torch.mean(torch.cat(discountedFutureRewardLosses))
        stateValueLoss = torch.mean(torch.cat(stateValueLosses))
        advantageLoss = torch.mean(torch.cat(advantageLosses))
        actionProbabilityLoss = torch.mean(torch.cat(actionProbabilityLosses))
        totalRewardLoss = presentRewardLoss + discountedFutureRewardLoss + stateValueLoss + advantageLoss + actionProbabilityLoss

        if self.agentConfiguration['enable_homogenization_loss']:
            targetHomogenizationLoss = torch.mean(torch.cat(targetHomogenizationLosses))
            extraLosses.append(targetHomogenizationLoss)
        else:
            targetHomogenizationLoss = zeroTensor

        if len(extraLosses) > 0:
            totalLoss = totalRewardLoss + torch.sum(torch.cat(extraLosses))
        else:
            totalLoss = totalRewardLoss

        self.optimizer.zero_grad()
        totalRebalancedLoss = None
        if not self.agentConfiguration['enable_loss_balancing']:
            totalLoss.backward()
        else:
            if len(self.trainingLosses['totalLoss']) > 2:
                averageStart = max(0, min(len(self.trainingLosses['totalRewardLoss']) - 1, self.agentConfiguration['loss_balancing_moving_average_period']))

                runningAverageRewardLoss = numpy.mean(self.trainingLosses['totalRewardLoss'][-averageStart:])
                runningAverageTracePredictionLoss = numpy.mean(self.trainingLosses['tracePredictionLoss'][-averageStart:])
                runningAverageExecutionFeaturesLoss = numpy.mean(self.trainingLosses['predictedExecutionFeaturesLoss'][-averageStart:])
                runningAverageHomogenizationLoss = numpy.mean(self.trainingLosses['targetHomogenizationLoss'][-averageStart:])
                runningAveragePredictedCursorLoss = numpy.mean(self.trainingLosses['predictedCursorLoss'][-averageStart:])

                tracePredictionAdustment = (runningAverageRewardLoss / (runningAverageTracePredictionLoss + 1e-6)) * self.agentConfiguration['loss_ratio_trace_prediction']
                executionFeaturesAdustment = (runningAverageRewardLoss / (runningAverageExecutionFeaturesLoss + 1e-6)) * self.agentConfiguration['loss_ratio_execution_features']
                homogenizationAdustment = (runningAverageRewardLoss / (runningAverageHomogenizationLoss + 1e-6)) * self.agentConfiguration['loss_ratio_homogenization']
                predictedCursorAdustment = (runningAverageRewardLoss / (runningAveragePredictedCursorLoss + 1e-6)) * self.agentConfiguration['loss_ratio_predicted_cursor']
            else:
                tracePredictionAdustment = 1
                executionFeaturesAdustment = 1
                homogenizationAdustment = 1
                predictedCursorAdustment = 1

            if not self.agentConfiguration['enable_trace_prediction_loss']:
                tracePredictionAdustment = 1

            if not self.agentConfiguration['enable_execution_feature_prediction_loss']:
                executionFeaturesAdustment = 1

            if not self.agentConfiguration['enable_cursor_prediction_loss']:
                predictedCursorAdustment = 1

            if not self.agentConfiguration['enable_homogenization_loss']:
                homogenizationAdustment = 1

            totalRebalancedLoss = totalRewardLoss + \
                                  tracePredictionLoss * self.variableWrapperFunc(torch.FloatTensor, [tracePredictionAdustment]) + \
                                  predictedExecutionFeaturesLoss * self.variableWrapperFunc(torch.FloatTensor, [executionFeaturesAdustment]) + \
                                  targetHomogenizationLoss * self.variableWrapperFunc(torch.FloatTensor, [homogenizationAdustment]) + \
                                  predictedCursorLoss * self.variableWrapperFunc(torch.FloatTensor, [predictedCursorAdustment])

            totalRebalancedLoss.backward()

        # Put a check in so that we don't do the optimizer step if there are NaNs in the loss
        if numpy.count_nonzero(numpy.isnan(float(totalLoss.data.item()))) == 0:
            self.optimizer.step()
        else:
            print(datetime.now(), "ERROR! NaN detected in loss calculation. Skipping optimization step.", flush=True)
            print(datetime.now(), "presentRewardLoss", float(presentRewardLoss.data.item()))
            print(datetime.now(), "discountedFutureRewardLoss", float(discountedFutureRewardLoss.data.item()))
            print(datetime.now(), "stateValueLoss", float(stateValueLoss.data.item()))
            print(datetime.now(), "advantageLoss", float(advantageLoss.data.item()))
            print(datetime.now(), "actionProbabilityLoss", float(actionProbabilityLoss.data.item()))
            print(datetime.now(), "tracePredictionLoss", float(tracePredictionLoss.data.item()))
            print(datetime.now(), "predictedExecutionFeaturesLoss", float(predictedExecutionFeaturesLoss.data.item()))
            print(datetime.now(), "targetHomogenizationLoss", float(targetHomogenizationLoss.data.item()))
            print(datetime.now(), "predictedCursorLoss", float(predictedCursorLoss.data.item()))

            return

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
        if self.agentConfiguration['enable_loss_balancing']:
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

        sampleRewardLosses = [tensor.data.item() for tensor in totalRewardLosses]

        return totalRewardLoss, presentRewardLoss, discountedFutureRewardLoss, stateValueLoss, advantageLoss, actionProbabilityLoss, tracePredictionLoss, predictedExecutionFeaturesLoss, targetHomogenizationLoss, predictedCursorLoss, totalLoss, totalRebalancedLoss, batchReward, sampleRewardLosses



def processRawImageParallel(rawImage):
    # shrunk = skimage.transform.resize(image, [int(width / 2), int(height / 2)])

    # Convert to grey-scale image
    processedImage = numpy.array([skimage.color.rgb2gray(rawImage[:, :, :3])])

    # Round the float values down to 0. This minimizes some the error introduced by the video codecs
    processedImage = numpy.around(processedImage, decimals=2)

    return processedImage
