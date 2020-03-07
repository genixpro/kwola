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
from kwola.models.actions.ClickTapAction import ClickTapAction
import os.path
import numpy
import skimage
import skimage.transform
import skimage.color

from .BradNet import BradNet

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs) if USE_CUDA else autograd.Variable(*args, **kwargs)

class DeepLearningAgent(BaseAgent):
    """
        This class represents a deep learning agent, which uses reinforcement learning to make the automated testing more effective
    """

    def __init__(self):
        pass

        self.num_frames = 1400000
        self.batch_size = 32
        self.gamma = 0.50
        self.frameStart = 7200

    def load(self):
        """
            Loads the agent from db / disk

            :return:
        """

        if os.path.exists("/home/bradley/kwola/deep_learning_model"):
            self.model.load_state_dict(torch.load("/home/bradley/kwola/deep_leraning_model"))

    def save(self):
        """
            Saves the agent to the db / disk.

            :return:
        """
        torch.save(self.model.state_dict(), "/home/bradley/deep_learning_model")


    def initialize(self, environment):
        """
        Initialize the agent for operating in the given environment.

        :param environment:
        :return:
        """
        self.environment = environment

        rect = environment.screenshotSize()

        self.model = BradNet([3, rect['height'], rect['width']], self.environment.branchFeatureSize())

        self.model = self.model.cuda()
        # self.model = self.model

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def getImage(self):
        image = cv2.imdecode(numpy.frombuffer(self.environment.driver.get_screenshot_as_png(), numpy.uint8), -1)

        width = image.shape[0]
        height = image.shape[1]

        # shrunk = skimage.transform.resize(image, [int(width / 2), int(height / 2)])

        # Convert to HSL representation, but discard the saturation layer
        image = skimage.color.rgb2hsv(image[:, :, :3])

        swapped = numpy.swapaxes(image, 0, 2)

        hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)

        return hueLightnessImage


    def getBranchFeature(self):
        feature = numpy.minimum(self.environment.lastCumulativeBranchExecutionVector, numpy.ones_like(self.environment.lastCumulativeBranchExecutionVector))
        return feature



    def nextBestAction(self):
        """
            Return the next best action predicted by the agent.
            # :param environment:
            :return:
        """
        epsilon = 0.5
        image = self.getImage()
        branchFeature = self.getBranchFeature()

        actionInfo = self.model.predict(image, branchFeature, epsilon)

        action = ClickTapAction(x=actionInfo[1][0], y=actionInfo[1][1])

        print("Click", action.x, action.y)

        return action



    def learnFromTestingSequence(self, testingSequence):
        """
            Runs the backward pass / gradient update so the algorithm can learn from all the memories in the given testing sequence

            :param testingSequence:
            :return:
        """

        frames = []

        cap = cv2.VideoCapture(f'/home/bradley/{str(testingSequence.id)}.mp4')

        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                image = skimage.color.rgb2hsv(frame[:, :, :3])

                swapped = numpy.swapaxes(image, 0, 2)

                hueLightnessImage = numpy.concatenate((swapped[0:1], swapped[2:]), axis=0)
                frames.append(hueLightnessImage)
            else:
                break

        # First compute the present reward at each time step
        present_rewards = []
        for trace in testingSequence.executionTraces:
            if trace.didActionSucceed == False:
                present_rewards.append(-1.0)
            elif trace.didNewBranchesExecute:
                present_rewards.append(1.0)
            else:
                present_rewards.append(0.0)

        discountRate = 0.98

        # Now compute the discounted reward
        discountedRewards = []
        present_rewards.reverse()
        current = 0
        for reward in present_rewards:
            current *= discountRate
            current += reward
            discountedRewards.append(current)

        discountedRewards.reverse()
        present_rewards.reverse()

        shuffledTraceFrameList = list(zip(testingSequence.executionTraces, frames, discountedRewards))
        random.shuffle(shuffledTraceFrameList)

        losses = []
        for trace, frame, discountedReward in shuffledTraceFrameList:
            branchFeature = numpy.array(trace.startCumulativeBranchExecutionTrace)

            frame = torch.unsqueeze(Variable(torch.Tensor(frame)), 0)

            q_values = self.model({"image": frame, "branchFeature": Variable(torch.FloatTensor(branchFeature), volatile=True) })

            width = frame.shape[3]
            height = frame.shape[2]

            q_values = q_values.reshape([-1, height * width])

            action_index = int(trace.actionPerformed.x * height + trace.actionPerformed.y)
            action_index = Variable(torch.LongTensor(numpy.array([action_index], dtype=numpy.int32)))

            action_index = action_index.unsqueeze(0)

            q_value = q_values.gather(1, action_index).squeeze(1)

            loss = (q_value - Variable(torch.FloatTensor(numpy.array([discountedReward], dtype=numpy.float32) ))).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            lossValue = float(loss.data.item())

            losses.append(lossValue)

        averageLoss = numpy.mean(losses)

        totalReward = float(numpy.sum(present_rewards))

        print(testingSequence.id)
        print("Total Reward", float(totalReward))
        print("Average Loss:", averageLoss)

