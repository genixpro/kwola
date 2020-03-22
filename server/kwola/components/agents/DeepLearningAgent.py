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


    def initialize(self, branchFeatureSize):
        """
        Initialize the agent.

        :return:
        """

        self.model = BradNet(self.agentConfiguration, branchFeatureSize * 2, len(self.actions), branchFeatureSize, 12, len(self.cursors), whichGpu=self.whichGpu)

        if self.whichGpu == "all":
            self.model = self.model.cuda()
        elif self.whichGpu is None:
            self.model = self.model.cpu()
        else:
            self.model = self.model.to(torch.device(f"cuda:{self.whichGpu}"))
        # self.model = self.model

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)


    def processImages(self, images):
        convertedImageFutures = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for image in images:
                convertedImageFuture = executor.submit(processRawImageParallel, image)
                convertedImageFutures.append(convertedImageFuture)

        converteHSLImages = [
            convertedImageFuture.result()[0] for convertedImageFuture in convertedImageFutures
        ]

        convertedSegmentationMaps = [
            convertedImageFuture.result()[1] for convertedImageFuture in convertedImageFutures
        ]

        return numpy.array(converteHSLImages),  numpy.array(convertedSegmentationMaps)


    def createPixelActionMap(self, actionMaps, height, width):
        pixelActionMap = numpy.zeros([3, height, width], dtype=numpy.uint8)

        for element in actionMaps:
            actionTypes = []

            if element['canClick']:
                actionTypes.append(self.actionsSorted.index("click"))
            if element['canType']:
                actionTypes.append(self.actionsSorted.index("typeEmail"))
                actionTypes.append(self.actionsSorted.index("typePassword"))

            for actionTypeIndex in actionTypes:
                pixelActionMap[actionTypeIndex, int(element['top']):int(element['bottom']), int(element['left']):int(element['right'])] = 1

        return pixelActionMap


    def nextBestActions(self, stepNumber, rawImages, envActionMaps, additionalFeatures):
        """
            Return the next best action predicted by the agent.

            :return:
        """
        processedImages, segmentationMaps = self.processImages(rawImages)
        actions = []

        width = processedImages.shape[3]
        height = processedImages.shape[2]

        batchSampleIndexes = []
        imageBatch = []
        additionalFeatureVectorBatch = []
        pixelActionMapsBatch = []
        segmentationMapsBatch = []
        epsilonsPerSample = []

        for sampleIndex, image, segmentationMap, additionalFeatureVector, actionMaps in zip(range(len(processedImages)), processedImages, segmentationMaps, additionalFeatures, envActionMaps):
            randomActionProbability = (float(sampleIndex + 1) / float(len(processedImages))) * 0.75 * (1 + (stepNumber / self.agentConfiguration['testing_sequence_length']))
            weightedRandomActionProbability = (float(sampleIndex + 2) / float(len(processedImages))) * 1.00 * (1 + (stepNumber / self.agentConfiguration['testing_sequence_length']))

            pixelActionMap = self.createPixelActionMap(actionMaps, height, width)

            if random.random() > randomActionProbability:
                batchSampleIndexes.append(sampleIndex)
                imageBatch.append(image)
                additionalFeatureVectorBatch.append(additionalFeatureVector)
                pixelActionMapsBatch.append(pixelActionMap)
                segmentationMapsBatch.append(segmentationMap)
                epsilonsPerSample.append(weightedRandomActionProbability)
            else:
                # Choose a random segment, but we only want to choose from among the segments that reside within
                # areas that have actions associated with them. We use the pixelActionMap to do that.
                segmentationMapMasked = numpy.array(segmentationMap + 1) * numpy.minimum(pixelActionMap.sum(axis=0), 1)
                uniqueSegmentsInsideMask = list(sorted(numpy.unique(segmentationMapMasked).tolist()))

                if len(uniqueSegmentsInsideMask) == 1:
                    print(datetime.now(), "Error! No segments to pick. Choosing random spot. This usually means there are no available actions on the tested application, such as a blank screen with no text, links, buttons or input elements, or that the action-mapping for this environment is not working for some reason.", flush=True)
                    actionX = random.randrange(0, width)
                    actionY = random.randrange(0, height)
                    actionType = random.randrange(0, len(self.actionsSorted))
                else:
                    chosenSegmentation = random.choice(uniqueSegmentsInsideMask[1:])
                    chosenPixel = self.getRandomPixelOfSegmentation(segmentationMapMasked, chosenSegmentation)

                    possibleActionsAtPixel = pixelActionMap[:, chosenPixel[1], chosenPixel[0]]
                    possibleActionIndexes = [actionIndex for actionIndex in range(len(self.actionsSorted)) if possibleActionsAtPixel[actionIndex]]

                    actionType = random.choice(possibleActionIndexes)
                    actionX = chosenPixel[0]
                    actionY = chosenPixel[1]

                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = "random"
                action.predictedReward = None
                actions.append((sampleIndex, action))

        if len(imageBatch) > 0:
            imageTensor = self.variableWrapperFunc(torch.FloatTensor(numpy.array(imageBatch)))
            additionalFeatureTensor = self.variableWrapperFunc(torch.FloatTensor(numpy.array(additionalFeatureVectorBatch)))
            pixelActionMapTensor = self.variableWrapperFunc(torch.FloatTensor(pixelActionMapsBatch))

            presentRewardPredictions, discountedFutureRewardPredictions, predictedTrace, predictedExecutionFeatures, predictedCursor, predictedPixelFeatures, stamp, actionProbabilities = self.model({
                "image": imageTensor,
                "additionalFeature": additionalFeatureTensor,
                "pixelActionMaps": pixelActionMapTensor
            })

            totalRewardPredictions = presentRewardPredictions + discountedFutureRewardPredictions

            for sampleIndex, segmentationMap, sampleEpsilon, sampleActionProbs, sampleRewardPredictions in zip(batchSampleIndexes, segmentationMapsBatch, epsilonsPerSample, actionProbabilities, totalRewardPredictions):
                if random.random() > sampleEpsilon:
                    source = "prediction"

                    actionType = sampleRewardPredictions.reshape([len(self.actionsSorted), width * height]).max(dim=1)[0].argmax(0).data.item()
                    actionY = sampleRewardPredictions[actionType].max(dim=1)[0].argmax(0).data.item()
                    actionX = sampleRewardPredictions[actionType, actionY].argmax(0).data.item()

                    samplePredictedReward = sampleRewardPredictions[actionType, actionY, actionX].data.item()
                else:
                    reshaped = numpy.array(sampleActionProbs.data).reshape([len(self.actionsSorted) * height * width])
                    actionIndex = numpy.random.choice(range(len(self.actionsSorted) * height * width), p=reshaped)
                    newProbs = numpy.zeros([len(self.actionsSorted) * height * width])
                    newProbs[actionIndex] = 1

                    newProbs = newProbs.reshape([len(self.actionsSorted), height * width])
                    actionType = newProbs.max(axis=1).argmax(axis=0)
                    newProbs = newProbs.reshape([len(self.actionsSorted), height, width])
                    actionY = newProbs[actionType].max(axis=1).argmax(axis=0)
                    actionX = newProbs[actionType, actionY].argmax(axis=0)


                    source = "weighted_random"
                    samplePredictedReward = sampleRewardPredictions[actionType, actionY, actionX].data.item()

                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = source
                action.predictedReward = samplePredictedReward
                actions.append((sampleIndex, action))

        sortedActions = sorted(actions, key=lambda row: row[0])

        return [action[1] for action in sortedActions]

    def getRandomPixelOfSegmentation(self, segmentationMap, chosenSegmentation):
        width = segmentationMap.shape[1]
        height = segmentationMap.shape[0]
        possiblePixels = []
        for x in range(width):
            for y in range(height):
                if segmentationMap[y, x] == chosenSegmentation:
                    possiblePixels.append((x, y))

        chosenPixel = random.choice(possiblePixels)
        return chosenPixel

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

                rewardChartAxes.set_ylim(ymin=-2.0, ymax=10.0)

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
                numColumns = 3
                numRows = 3
                boundaryImageShrinkRatio = 3

                mainColorMap = plt.get_cmap('inferno')
                greyColorMap = plt.get_cmap('gray')

                mainFigure = plt.figure(
                    figsize=((rightSize) / 100, (imageHeight + bottomSize + topSize - chartTopMargin) / 100), dpi=100)

                rewardPredictionAxes = [
                    mainFigure.add_subplot(numColumns, numRows, actionIndex + 1)
                    for actionIndex, action in enumerate(self.actionsSorted)
                ]

                stampAxes = mainFigure.add_subplot(numColumns, numRows, len(self.actionsSorted) + 1)

                processedImage, segmentationMap = processRawImageParallel(rawImage)
                boundaryImage = skimage.segmentation.mark_boundaries(rawImage[::boundaryImageShrinkRatio, ::boundaryImageShrinkRatio], segmentationMap[::boundaryImageShrinkRatio, ::boundaryImageShrinkRatio])

                segmentationBoundaryAxes = mainFigure.add_subplot(numColumns, numRows, len(self.actionsSorted) + 2)
                segmentationBoundaryAxes.imshow(boundaryImage, vmin=0, vmax=1, interpolation="bilinear")
                segmentationBoundaryAxes.set_xticks([])
                segmentationBoundaryAxes.set_yticks([])
                segmentationBoundaryAxes.set_title(f"{len(numpy.unique(segmentationMap))} segments")

                rewardPixelMaskAxes = mainFigure.add_subplot(numColumns, numRows, len(self.actionsSorted) + 3)
                rewardPixelMask = self.createRewardPixelMask(processedImage, trace)
                rewardPixelCount = numpy.count_nonzero(rewardPixelMask)
                rewardPixelMaskAxes.imshow(rewardPixelMask, vmin=0, vmax=1, cmap=plt.get_cmap("gray"), interpolation="bilinear")
                rewardPixelMaskAxes.set_xticks([])
                rewardPixelMaskAxes.set_yticks([])
                rewardPixelMaskAxes.set_title(f"{rewardPixelCount} target pixels")

                pixelActionMapAxes = mainFigure.add_subplot(numColumns, numRows, len(self.actionsSorted) + 4)
                pixelActionMap = self.createPixelActionMap(trace.actionMaps, imageHeight, imageWidth)
                actionPixelCount = numpy.count_nonzero(pixelActionMap)
                pixelActionMapAxes.imshow(numpy.swapaxes(numpy.swapaxes(pixelActionMap, 0, 1), 1, 2) * 255, interpolation="bilinear")
                pixelActionMapAxes.set_xticks([])
                pixelActionMapAxes.set_yticks([])
                pixelActionMapAxes.set_title(f"{actionPixelCount} action pixels")

                additionalFeature = self.prepareAdditionalFeaturesForTrace(trace)

                presentRewardPredictions, discountedFutureRewardPredictions, predictedTrace, predictedExecutionFeatures, predictedCursor, predictedPixelFeatures, stamp, actionProbabilities = \
                    self.model({"image": self.variableWrapperFunc(torch.FloatTensor(numpy.array([processedImage]))),
                                "additionalFeature": self.variableWrapperFunc(torch.FloatTensor(additionalFeature)),
                                "pixelActionMaps": self.variableWrapperFunc(torch.FloatTensor(numpy.array([pixelActionMap])))
                                })
                totalRewardPredictions = numpy.array((presentRewardPredictions + discountedFutureRewardPredictions).data)


                for actionIndex, action in enumerate(self.actionsSorted):
                    actionY = totalRewardPredictions[0][actionIndex].max(axis=1).argmax(axis=0)
                    actionX = totalRewardPredictions[0][actionIndex, actionY].argmax(axis=0)

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

                    rewardPredictionsShrunk = skimage.measure.block_reduce(totalRewardPredictions[0][actionIndex], (3,3), numpy.max)

                    im = rewardPredictionAxes[actionIndex].imshow(rewardPredictionsShrunk, cmap=mainColorMap,
                                                                  vmin=-0.50, vmax=0.75, interpolation="nearest")
                    rewardPredictionAxes[actionIndex].set_title(f"{action} {maxValue:.3f}")
                    mainFigure.colorbar(im, ax=rewardPredictionAxes[actionIndex], orientation='vertical')

                stampAxes.set_xticks([])
                stampAxes.set_yticks([])
                stampIm = stampAxes.imshow(stamp.data[0], cmap=greyColorMap, interpolation="nearest", vmin=-20.0, vmax=20.0)
                mainFigure.colorbar(stampIm, ax=stampAxes, orientation='vertical')
                stampAxes.set_title("Memory Stamp")

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


    def createRewardPixelMask(self, processedImage, trace):
        # We use flood-segmentation on the original image to select which pixels we will update reward values for.
        # This works great on UIs because the elements always have big areas of solid-color which respond in the same
        # way.
        rewardPixelMask = skimage.segmentation.flood(processedImage[0], (int(trace.actionPerformed.y), int(trace.actionPerformed.x)))

        return rewardPixelMask


    def prepareAdditionalFeaturesForTrace(self, trace):
        branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
        decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
        additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

        return additionalFeature


    def calculateTrainingCropPosition(self, trace, imageWidth, imageHeight):
        cropLeft = trace.actionPerformed.x - self.agentConfiguration['training_crop_width'] / 2
        cropTop = trace.actionPerformed.y - self.agentConfiguration['training_crop_height'] / 2

        cropLeft = max(0, cropLeft)
        cropLeft = min(imageWidth - self.agentConfiguration['training_crop_width'], cropLeft)

        cropTop = max(0, cropTop)
        cropTop = min(imageHeight - self.agentConfiguration['training_crop_height'], cropTop)

        cropRight = cropLeft + self.agentConfiguration['training_crop_width']
        cropBottom = cropTop + self.agentConfiguration['training_crop_height']

        return (int(cropLeft), int(cropTop), int(cropRight), int(cropBottom))


    @staticmethod
    def computeTotalRewardsParallel(execusionSessionId):
        session = ExecutionSession.objects(id=bson.ObjectId(execusionSessionId)).first()

        presentRewards = DeepLearningAgent.computePresentRewards(session)
        futureRewards = DeepLearningAgent.computeDiscountedFutureRewards(session)

        return [present + future for present, future in zip(presentRewards, futureRewards)]

    @staticmethod
    def createTrainingRewardNormalizer(execusionSessionIds):
        rewardFutures = []

        rewardSequences = []
        longest = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            for sessionId in execusionSessionIds:
                rewardFutures.append(executor.submit(DeepLearningAgent.computeTotalRewardsParallel, str(sessionId)))

            for rewardFuture in concurrent.futures.as_completed(rewardFutures):
                totalRewards = rewardFuture.result()
                rewardSequences.append(totalRewards)
                longest = max(longest, len(rewardSequences))

        for totalRewards in rewardSequences:
            while len(totalRewards) < longest:
                # This isn't the greatest solution for handling execution sessions with different lengths, but it works
                # for now when we pretty much always have execution sessions of the same length except when debugging.
                totalRewards.append(0)

        rewardSequences = numpy.array(rewardSequences)
        trainingRewardNormalizer = sklearn.preprocessing.StandardScaler().fit(rewardSequences)

        return trainingRewardNormalizer



    def prepareBatchesForExecutionSession(self, executionSession, trainingRewardNormalizer=None):
        """
            This function prepares batches that can be fed to the neural network.

            :param executionSession:
            :param , trainingRewardNormalizer
            :return:
        """
        processedImages = []

        videoPath = config.getKwolaUserDataDirectory("videos")
        for rawImage in DeepLearningAgent.readVideoFrames(os.path.join(videoPath, f'{str(executionSession.id)}.mp4')):
            processedImage = processRawImageParallel(rawImage, doSegmentation=False)
            processedImages.append(processedImage)

        # First compute the present reward at each time step
        presentRewards = DeepLearningAgent.computePresentRewards(executionSession)

        # Now compute the discounted reward
        discountedFutureRewards = DeepLearningAgent.computeDiscountedFutureRewards(executionSession)

        # If there is a normalizer, use if to normalize the rewards
        if trainingRewardNormalizer is not None:
            if len(presentRewards) < len(trainingRewardNormalizer.mean_):
               presentRewardsNormalizationInput = presentRewards + ([1] * (len(trainingRewardNormalizer.mean_) - len(presentRewards)))
               discountedFutureRewardsNormalizationInput = discountedFutureRewards + ([1] * (len(trainingRewardNormalizer.mean_) - len(presentRewards)))
            else:
                presentRewardsNormalizationInput = presentRewards
                discountedFutureRewardsNormalizationInput = discountedFutureRewards

            # Compute the total rewards and normalize
            totalRewardsNormalizationInput = numpy.array([presentRewardsNormalizationInput]) + numpy.array([discountedFutureRewardsNormalizationInput])
            normalizedTotalRewards = trainingRewardNormalizer.transform(totalRewardsNormalizationInput)[0]

            normalizedPresentRewards = [
                (abs(present) / (abs(present) + abs(future) + 0.01)) * normalized for present, future, normalized
                in zip(presentRewards, discountedFutureRewards, normalizedTotalRewards[:len(presentRewards)])
            ]

            normalizedDiscountedFutureRewards = [
                (abs(future) / (abs(present) + abs(future) + 0.01)) * normalized for present, future, normalized
                in zip(presentRewards, discountedFutureRewards, normalizedTotalRewards[:len(presentRewards)])
            ]

            presentRewards = normalizedPresentRewards
            discountedFutureRewards = normalizedDiscountedFutureRewards

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

        shuffledTraceFrameList = list(zip(executionSession.executionTraces, processedImages, discountedFutureRewards, presentRewards, executionTraces))
        random.shuffle(shuffledTraceFrameList)

        batches = []

        for batch in grouper(self.agentConfiguration['batch_size'], shuffledTraceFrameList):
            batchProcessedImages = []
            batchAdditionalFeatures = []
            batchPixelActionMaps = []
            batchActionTypes = []
            batchActionXs = []
            batchActionYs = []
            batchExecutionTraces = []
            batchDiscountedFutureRewards = []
            batchPresentRewards = []
            batchRewardPixelMasks = []
            batchExecutionFeatures = []
            batchCursors = []

            for trace, processedImage, discountedFutureReward, presentReward, executionTrace in batch:
                width = processedImage.shape[2]
                height = processedImage.shape[1]

                processedImage = processedImage + numpy.random.normal(loc=0, scale=self.agentConfiguration['training_image_gaussian_noise_scale'], size=processedImage.shape)

                cropLeft, cropTop, cropRight, cropBottom = self.calculateTrainingCropPosition(trace, imageWidth=width, imageHeight=height)

                branchFeature = numpy.minimum(trace.startCumulativeBranchExecutionTrace, numpy.ones_like(trace.startCumulativeBranchExecutionTrace))
                decayingExecutionTraceFeature = numpy.array(trace.startDecayingExecutionTrace)
                additionalFeature = numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=0)

                batchProcessedImages.append(processedImage[:, cropTop:cropBottom, cropLeft:cropRight])
                batchAdditionalFeatures.append(additionalFeature)

                pixelActionMap = self.createPixelActionMap(trace.actionMaps, height, width)
                batchPixelActionMaps.append(pixelActionMap[:, cropTop:cropBottom, cropLeft:cropRight])

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

                batchDiscountedFutureRewards.append(discountedFutureReward)
                batchPresentRewards.append(presentReward)

                rewardPixelMask = self.createRewardPixelMask(processedImage, trace)

                batchRewardPixelMasks.append(rewardPixelMask[cropTop:cropBottom, cropLeft:cropRight])

                batchExecutionFeatures.append(executionFeatures)

            # Append an array with all the data to the list of batches.
            # Add the same time we down-sample some of the data points to be more compact.
            # We don't need a high precision for the image itself
            batches.append({
                "processedImages": numpy.array(batchProcessedImages, dtype=numpy.float16),
                "additionalFeatures": numpy.array(batchAdditionalFeatures, dtype=numpy.float16),
                "pixelActionMaps": numpy.array(batchPixelActionMaps, dtype=numpy.uint8),
                "actionTypes": batchActionTypes,
                "actionXs": numpy.array(batchActionXs, dtype=numpy.int16),
                "actionYs": numpy.array(batchActionYs, dtype=numpy.int16),
                "executionTraces": numpy.array(batchExecutionTraces, dtype=numpy.int8),
                "discountedFutureRewards": numpy.array(batchDiscountedFutureRewards, dtype=numpy.float32),
                "presentRewards": numpy.array(batchPresentRewards, dtype=numpy.float32),
                "rewardPixelMasks": numpy.array(batchRewardPixelMasks, dtype=numpy.uint8),
                "executionFeatures": numpy.array(batchExecutionFeatures, dtype=numpy.uint8),
                "cursors": numpy.array(batchCursors, dtype=numpy.uint8)
            })
            # print(datetime.now(), "Finished preparing batch #", len(batches), "for", str(executionSession.id), flush=True)

        # print(datetime.now(), "Finished preparing all batches for", str(executionSession.id), flush=True)

        return batches

    def learnFromBatch(self, batch):
        """
            Runs backprop on the neural network with the given batch.

            :param batch: A batch of image/action/output pairs. Should be the return value from prepareBatchesForTestingSequence
            :return:
        """
        presentRewardPredictions, discountedFutureRewardPredictions, predictedTraces, predictedExecutionFeatures, predictedCursors, predictedPixelFeatures, stamp, actionProbabilities = self.model({
            "image": self.variableWrapperFunc(torch.FloatTensor(numpy.array(batch['processedImages']))),
            "additionalFeature": self.variableWrapperFunc(torch.FloatTensor(batch['additionalFeatures'])),
            "pixelActionMaps": self.variableWrapperFunc(torch.FloatTensor(batch['pixelActionMaps'])),
            "action_type": batch['actionTypes'],
            "action_x": batch['actionXs'],
            "action_y": batch['actionYs']
        })

        totalRewardLosses = []
        presentRewardLosses = []
        targetHomogenizationLosses = []
        discountedFutureRewardLosses = []

        for presentRewardImage, discountedFutureRewardImage, pixelFeatureImage, rewardPixelMask, presentReward, discountedFutureReward, actionType, actionX, actionY, pixelActionMap in zip(presentRewardPredictions, discountedFutureRewardPredictions, predictedPixelFeatures, batch['rewardPixelMasks'], batch['presentRewards'], batch['discountedFutureRewards'], batch['actionTypes'], batch['actionXs'], batch['actionYs'], batch['pixelActionMaps']):
            # if len(totalRewardLosses) == 0:
            #     cv2.imshow('image', rewardPixelMask * 200)
            #     cv2.waitKey(50)
            width = presentRewardImage.shape[2]
            height = presentRewardImage.shape[1]

            rewardPixelMask = self.variableWrapperFunc(torch.IntTensor(rewardPixelMask))
            pixelActionMap = self.variableWrapperFunc(torch.IntTensor(pixelActionMap))
            # actionType = self.variableWrapperFunc(torch.IntTensor(actionType))

            presentRewardsMasked = presentRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask
            discountedFutureRewardsMasked = discountedFutureRewardImage[self.actionsSorted.index(actionType)] * rewardPixelMask

            torchBatchPresentRewards = torch.ones_like(presentRewardImage[self.actionsSorted.index(actionType)]) * self.variableWrapperFunc(torch.FloatTensor([presentReward])) * rewardPixelMask
            torchBatchDiscountedFutureRewards = torch.ones_like(presentRewardImage[self.actionsSorted.index(actionType)]) * self.variableWrapperFunc(torch.FloatTensor([discountedFutureReward])) * rewardPixelMask

            countPixelMask = (rewardPixelMask.sum())

            presentRewardLossMap = (torchBatchPresentRewards - presentRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]
            discountedFutureRewardLossMap = (torchBatchDiscountedFutureRewards - discountedFutureRewardsMasked) * pixelActionMap[self.actionsSorted.index(actionType)]

            presentRewardLoss = presentRewardLossMap.pow(2).sum() / countPixelMask
            discountedFutureRewardLoss = discountedFutureRewardLossMap.pow(2).sum() / countPixelMask

            if self.agentConfiguration['enable_homogenization_loss']:
                # Target Homogenization loss - basically, all of the pixels for the masked area should produce similar features to each other and different features
                # from other pixels.
                targetHomogenizationDifferenceMap = ((pixelFeatureImage - pixelFeatureImage[:, actionY, actionX].unsqueeze(1).unsqueeze(1)) * rewardPixelMask).pow(2).mean(axis=0)
                targetDifferentiationDifferenceMap = ((pixelFeatureImage - pixelFeatureImage[:, actionY, actionX].unsqueeze(1).unsqueeze(1)) * (1.0 - rewardPixelMask)).pow(2).mean(axis=0)
                targetHomogenizationLoss = torch.abs((targetHomogenizationDifferenceMap.sum() / countPixelMask) - (targetDifferentiationDifferenceMap.sum() / (width * height - countPixelMask)))
                targetHomogenizationLosses.append(targetHomogenizationLoss.unsqueeze(0))

            # if len(totalRewardLosses) == 0:
            #     showRewardImageDebug(numpy.array(presentRewardsMasked.cpu().data), 'present-reward-prediction-masked', vmin=-10, vmax=20)
            #     showRewardImageDebug(numpy.array(discountedFutureRewardsMasked.cpu().data), 'future-reward-prediction-masked', vmin=-10, vmax=20)
            #
            #     showRewardImageDebug(numpy.array(torchBatchPresentRewards.cpu().data), 'present-reward-target', vmin=-10, vmax=20)
            #     showRewardImageDebug(numpy.array(torchBatchDiscountedFutureRewards.cpu().data), 'future-reward-target', vmin=-10, vmax=20)
            #
            #     showRewardImageDebug(numpy.array(presentRewardLossMap.cpu().data), 'present-loss', vmin=-10, vmax=20)
            #     showRewardImageDebug(numpy.array(discountedFutureRewardLossMap.cpu().data), "future-loss", vmin=-10, vmax=20)

                # showRewardImageDebug(numpy.array(targetHomogenizationDifferenceMap.cpu().data), 'homogenization')
                # showRewardImageDebug(numpy.array(targetDifferentiationDifferenceMap.cpu().data), 'differentiation')
                # cv2.waitKey(50)

            sampleLoss = presentRewardLoss + discountedFutureRewardLoss
            totalRewardLosses.append(sampleLoss.unsqueeze(0))

            presentRewardLosses.append(presentRewardLoss.unsqueeze(0))
            discountedFutureRewardLosses.append(discountedFutureRewardLoss.unsqueeze(0))

        if self.agentConfiguration['enable_trace_prediction_loss']:
            tracePredictionLoss = (predictedTraces - self.variableWrapperFunc(torch.FloatTensor(batch['executionTraces']))).abs().mean()
        else:
            tracePredictionLoss = self.variableWrapperFunc(torch.FloatTensor([0]))


        if self.agentConfiguration['enable_execution_feature_prediction_loss']:
            predictedExecutionFeaturesLoss = (predictedExecutionFeatures - self.variableWrapperFunc(torch.FloatTensor(batch['executionFeatures']))).abs().mean()
        else:
            predictedExecutionFeaturesLoss = self.variableWrapperFunc(torch.FloatTensor([0]))


        if self.agentConfiguration['enable_cursor_prediction_loss']:
            predictedCursorLoss = (predictedCursors - self.variableWrapperFunc(torch.FloatTensor(batch['cursors']))).abs().mean()
        else:
            predictedCursorLoss = self.variableWrapperFunc(torch.FloatTensor([0]))

        totalRewardLoss = torch.mean(torch.cat(totalRewardLosses))
        presentRewardLoss = torch.mean(torch.cat(presentRewardLosses))
        discountedFutureRewardLoss = torch.mean(torch.cat(discountedFutureRewardLosses))

        if self.agentConfiguration['enable_homogenization_loss']:
            targetHomogenizationLoss = torch.mean(torch.cat(targetHomogenizationLosses))
        else:
            targetHomogenizationLoss = self.variableWrapperFunc(torch.FloatTensor([0]))

        totalLoss = totalRewardLoss + tracePredictionLoss + predictedExecutionFeaturesLoss + targetHomogenizationLoss + predictedCursorLoss

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
                              tracePredictionLoss * self.variableWrapperFunc(torch.FloatTensor([tracePredictionAdustment])) + \
                              predictedExecutionFeaturesLoss * self.variableWrapperFunc(torch.FloatTensor([executionFeaturesAdustment])) + \
                              targetHomogenizationLoss * self.variableWrapperFunc(torch.FloatTensor([homogenizationAdustment])) + \
                              predictedCursorLoss * self.variableWrapperFunc(torch.FloatTensor([predictedCursorAdustment]))

        if numpy.count_nonzero(numpy.isnan(float(totalLoss.data.item()))) == 0:
            self.optimizer.zero_grad()
            if self.agentConfiguration['enable_loss_balancing']:
                totalRebalancedLoss.backward()
            else:
                totalLoss.backward()
            self.optimizer.step()
        else:
            print("ERROR! NaN detected in loss calculation. Skipping backward pass.", flush=True)

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



def globalSegmentImage(rawImage):
    # Segment the image. We use this to help the algorithm with randomly choosing
    # sections
    segmentationMap = felzenszwalb(rawImage, scale=300, sigma=0.50, min_size=70)
    return segmentationMap



def processRawImageParallel(rawImage, doSegmentation=True):
    # shrunk = skimage.transform.resize(image, [int(width / 2), int(height / 2)])

    # Convert to grey-scale image
    processedImage = numpy.array([skimage.color.rgb2gray(rawImage[:, :, :3])])

    # Round the float values down to 0. This minimizes some the error introduced by the video codecs
    processedImage = numpy.around(processedImage, decimals=2)

    if doSegmentation:
        segmentationMap = globalSegmentImage(rawImage)

        return processedImage, segmentationMap
    else:
        return processedImage
