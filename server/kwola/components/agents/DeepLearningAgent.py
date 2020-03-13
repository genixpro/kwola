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
import subprocess
import bz2
import os.path
from kwola.models.actions.ClickTapAction import ClickTapAction
import os.path
import numpy
import skimage
import skimage.draw
import skimage.transform
import skimage.color
import skimage.io
from skimage.segmentation import felzenszwalb, mark_boundaries
import shutil
import cv2
import matplotlib.pyplot as plt
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

        converteHSLImages = [
            convertedImageFuture.result()[0] for convertedImageFuture in convertedImageFutures
        ]

        convertedSegmentationMaps = [
            convertedImageFuture.result()[1] for convertedImageFuture in convertedImageFutures
        ]

        return numpy.array(converteHSLImages),  numpy.array(convertedSegmentationMaps)

    def getAdditionalFeatures(self):
        branchFeature = self.environment.getBranchFeatures()
        decayingExecutionTraceFeature = self.environment.getExecutionTraceFeatures()

        return numpy.concatenate([branchFeature, decayingExecutionTraceFeature], axis=1)

    def nextBestActions(self, stepNumber):
        """
            Return the next best action predicted by the agent.

            # :param environment:
            :return:
        """
        images, segmentationMaps = self.getImage()
        additionalFeatures = self.getAdditionalFeatures()
        actions = []

        width = images.shape[3]
        height = images.shape[2]

        for sampleN, image, segmentationMap, additionalFeatureVector in zip(range(len(images)), images, segmentationMaps, additionalFeatures):
            epsilon = (float(sampleN + 1) / float(len(images))) * 0.85 * (1 + (stepNumber / self.agentConfiguration['testing_sequence_length']))

            if random.random() > epsilon:
                image = self.variableWrapperFunc(torch.FloatTensor(numpy.array([image])))
                additionalFeatureVector = self.variableWrapperFunc(torch.FloatTensor(numpy.array([additionalFeatureVector])))

                presentRewardPredictions, discountedFutureRewardPredictions, predictedTrace, predictedExecutionFeatures, predictedCursor, predictedPixelFeatures = self.model({"image": image, "additionalFeature": additionalFeatureVector})

                totalRewardPredictions = presentRewardPredictions + discountedFutureRewardPredictions

                actionIndexes = totalRewardPredictions.reshape([1, width * height * len(self.actionsSorted)]).argmax(1).data

                actionType, actionX, actionY = BradNet.actionIndexToActionDetails(width, height, len(self.actionsSorted), actionIndexes[0])

                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = "prediction"
                actions.append(action)

            else:
                uniques = numpy.unique(segmentationMap)
                chosenSegmentation = random.choice(uniques)

                possiblePixels = []
                for x in range(width):
                    for y in range(height):
                        if segmentationMap[y, x] == chosenSegmentation:
                            possiblePixels.append((x, y))

                chosenPixel = random.choice(possiblePixels)

                actionType = random.randrange(0, len(self.actionsSorted))
                actionX = chosenPixel[0]
                actionY = chosenPixel[1]

                action = self.actions[self.actionsSorted[actionType]](actionX, actionY)
                action.source = "random"
                actions.append(action)

        return actions

    def computePresentRewards(self, executionSession):
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
        return presentRewards

    def computeDiscountedFutureRewards(self, executionSession):
        # First compute the present reward at each time step
        presentRewards = self.computePresentRewards(executionSession)

        # Now compute the discounted reward
        discountedFutureRewards = []
        presentRewards.reverse()
        current = 0
        for reward in presentRewards:
            current *= self.agentConfiguration['reward_discount_rate']
            discountedFutureRewards.append(current)
            current += reward

        discountedFutureRewards.reverse()

        return discountedFutureRewards

    def createDebugVideoForExecutionSession(self, executionSession):
        videoPath = config.getKwolaUserDataDirectory("videos")
        cap = cv2.VideoCapture(os.path.join(videoPath, f'{str(executionSession.id)}.mp4'))

        frames = []

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                frames.append(frame)
            else:
                break

        presentRewards = self.computePresentRewards(executionSession)

        discountedFutureRewards = self.computeDiscountedFutureRewards(executionSession)

        tempScreenshotDirectory = tempfile.mkdtemp()

        topSize = 200
        bottomSize = 250
        leftSize = 100
        rightSize = 100
        topMargin = 25

        frameHeight = frames[0].shape[0]
        frameWidth = frames[0].shape[1]

        newFrameIndex = 0

        lastFrame = frames.pop(0)
        lastFrame = numpy.flip(lastFrame, axis=2)

        def addDebugCircleToImage(image, trace):
            targetCircleCoordsRadius30 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y), int(leftSize + trace.actionPerformed.x), 30, shape=[int(frameWidth + extraWidth), int(frameHeight + extraHeight)])
            targetCircleCoordsRadius20 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y), int(leftSize + trace.actionPerformed.x), 20, shape=[int(frameWidth + extraWidth), int(frameHeight + extraHeight)])
            targetCircleCoordsRadius10 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y), int(leftSize + trace.actionPerformed.x), 10, shape=[int(frameWidth + extraWidth), int(frameHeight + extraHeight)])
            targetCircleCoordsRadius5 = skimage.draw.circle_perimeter(int(topSize + trace.actionPerformed.y), int(leftSize + trace.actionPerformed.x), 5, shape=[int(frameWidth + extraWidth), int(frameHeight + extraHeight)])
            image[targetCircleCoordsRadius30] = [255, 0, 0]
            image[targetCircleCoordsRadius20] = [255, 0, 0]
            image[targetCircleCoordsRadius10] = [255, 0, 0]
            image[targetCircleCoordsRadius5] = [255, 0, 0]

        def addDebugTextToImage(image, trace):
            fontSize = 0.5
            fontThickness = 1
            fontColor = (0, 0, 0)

            cv2.putText(image, f"URL {trace.startURL}", (leftSize-0, topMargin + 20), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"{str(executionSession.id)}", (leftSize-0, topMargin + 40), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"Frame {trace.frameNumber}", (leftSize-0, topMargin + 60), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"Action {trace.actionPerformed.type} at {trace.actionPerformed.x},{trace.actionPerformed.y}", (leftSize-0, topMargin + 80), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"Succeed: {str(trace.didActionSucceed)}", (leftSize-0, topMargin + 100), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"Error: {str(trace.didErrorOccur)}", (leftSize-0, topMargin + 120), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"New Error: {str(trace.didNewErrorOccur)}", (leftSize-0, topMargin + 140), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"Code Execute: {str(trace.didCodeExecute)}", (leftSize+300, topMargin + 40), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"New Branches: {str(trace.didNewBranchesExecute)}", (leftSize+300, topMargin + 60), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"Network Traffic: {str(trace.hadNetworkTraffic)}", (leftSize+300, topMargin + 80), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"New Network Traffic: {str(trace.hadNewNetworkTraffic)}", (leftSize+300, topMargin + 100), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"Screenshot Change: {str(trace.didScreenshotChange)}", (leftSize+300, topMargin + 120), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"New Screenshot: {str(trace.isScreenshotNew)}", (leftSize+300, topMargin + 140), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"URL Change: {str(trace.didURLChange)}", (leftSize+550, topMargin + 40), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"New URL: {str(trace.isURLNew)}", (leftSize+550, topMargin + 60), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"Had Log Output: {trace.hadLogOutput}", (leftSize+550, topMargin + 80), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)

            cv2.putText(image, f"Branch Coverage: {(trace.cumulativeBranchCoverage * 100):.2f}%", (leftSize+550, topMargin + 100), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"Present Reward: {(presentReward):.6f}", (leftSize+550, topMargin + 120), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)
            cv2.putText(image, f"Discounted Future Reward: {(discountedFutureReward):.6f}", (leftSize+550, topMargin + 140), cv2.FONT_HERSHEY_SIMPLEX, fontSize, fontColor, fontThickness, cv2.LINE_AA)



        fig = plt.figure(figsize=(frameWidth / 100, (bottomSize - 50) / 100), dpi=100)
        ax = fig.add_subplot(111)

        xCoords = numpy.array(range(len(presentRewards)))

        ax.set_ylim(ymin=0.0, ymax=0.7)

        ax.plot(xCoords, presentRewards)
        ax.plot(xCoords, discountedFutureRewards)

        ax.set_xticks(range(0, len(presentRewards), 5))
        ax.set_xticklabels([str(n) for n in range(0, len(presentRewards), 5)])
        ax.set_yticks(numpy.arange(0, 1, 1.0))
        ax.set_yticklabels(["" for n in range(2)])

        # ax.grid()
        fig.tight_layout()

        def addRewardChartToImage(image, trace):
            ax.set_xlim(xmin=trace.frameNumber - 20, xmax=trace.frameNumber + 20)
            line = ax.axvline(trace.frameNumber - 1, color='black', linewidth=2)

            # If we haven't already shown or saved the plot, then we need to
            # draw the figure first...
            fig.canvas.draw()

            # Now we can save it to a numpy array.
            rewardChart = numpy.fromstring(fig.canvas.tostring_rgb(), dtype=numpy.uint8, sep='')
            rewardChart = rewardChart.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            image[topSize + frameHeight:-50, leftSize:-rightSize] = rewardChart

            line.remove()


        for frameIndex, trace, frame, presentReward, discountedFutureReward, in zip(range(len(frames)), executionSession.executionTraces, frames, presentRewards, discountedFutureRewards):
            frame = numpy.flip(frame, axis=2)

            extraWidth = leftSize + rightSize
            extraHeight = topSize + bottomSize

            newImage = numpy.ones([frameHeight + extraHeight, frameWidth + extraWidth, 3]) * 255
            newImage[topSize:-bottomSize, leftSize:-rightSize] = lastFrame
            addDebugTextToImage(newImage, trace)
            addDebugCircleToImage(newImage, trace)
            addRewardChartToImage(newImage, trace)

            fileName = f"kwola-screenshot-{newFrameIndex:05d}.png"
            filePath = os.path.join(tempScreenshotDirectory, fileName)
            skimage.io.imsave(filePath, newImage)
            newFrameIndex += 1

            newImage = numpy.ones([frameHeight + extraHeight, frameWidth + extraWidth, 3]) * 255
            addDebugTextToImage(newImage, trace)

            newImage[topSize:-bottomSize, leftSize:-rightSize] = frame
            addRewardChartToImage(newImage, trace)

            fileName = f"kwola-screenshot-{newFrameIndex:05d}.png"
            filePath = os.path.join(tempScreenshotDirectory, fileName)
            skimage.io.imsave(filePath, numpy.array(newImage, dtype=numpy.uint8))
            newFrameIndex += 1

            lastFrame = frame

        subprocess.run(['ffmpeg', '-r', '60', '-f', 'image2', "-r", "2", '-i', 'kwola-screenshot-%05d.png', '-vcodec', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', "debug.mp4"], cwd=tempScreenshotDirectory)

        moviePath = os.path.join(tempScreenshotDirectory, "debug.mp4")

        with open(moviePath, "rb") as file:
            videoData = file.read()

        shutil.rmtree(tempScreenshotDirectory)

        return videoData



    def prepareBatchesForExecutionSession(self, testingSequence, executionSession):
        """
            This function prepares batches that can be fed to the neural network.

            :param testingSequence:
            :param executionSession:
            :return:
        """
        frames = []

        videoPath = config.getKwolaUserDataDirectory("videos")
        cap = cv2.VideoCapture(os.path.join(videoPath, f'{str(executionSession.id)}.mp4'))

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
        presentRewards = self.computePresentRewards(executionSession)

        # Now compute the discounted reward
        discountedFutureRewards = self.computeDiscountedFutureRewards(executionSession)

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

    # Segment the image, and randomly choose a segment. This helps to avoid wasting time clicking on useless
    # parts of the UI.
    segmentationMap = felzenszwalb(rawImage, scale=300, sigma=0.50, min_size=70)

    return hueLightnessImage, segmentationMap
