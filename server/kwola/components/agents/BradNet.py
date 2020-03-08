import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import scipy.signal
import pandas
from torchvision import models

# USE_CUDA = torch.cuda.is_available()
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs) if USE_CUDA else autograd.Variable(*args, **kwargs)

Variable = lambda x:x.cuda()


class BradNet(nn.Module):
    def __init__(self, additionalFeatureSize, numActions, executionTracePredictorSize):
        super(BradNet, self).__init__()

        self.branchStampEdgeSize = 10

        self.stampProjection = nn.Linear(additionalFeatureSize, self.branchStampEdgeSize*self.branchStampEdgeSize)

        self.pixelFeatureCount = 32

        self.innerSize = 64

        self.peakInnerSize = 128

        # self.mainModel = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=self.pixelFeatureCount)
        self.mainModel = nn.Sequential(
            nn.Conv2d(3, self.innerSize, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.innerSize),

            nn.Conv2d(self.innerSize, self.innerSize, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.innerSize),

            nn.Conv2d(self.innerSize, self.peakInnerSize, kernel_size=3, stride=2, dilation=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.peakInnerSize),

            nn.Conv2d(self.peakInnerSize, self.innerSize, kernel_size=5, stride=1, dilation=3, padding=6),
            nn.ReLU(),
            nn.BatchNorm2d(self.innerSize),

            nn.Conv2d(self.innerSize, self.pixelFeatureCount, kernel_size=5, stride=1, dilation=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.pixelFeatureCount),

            torch.nn.Upsample(scale_factor=8)
        )

        self.rewardConvolution = nn.Conv2d(self.pixelFeatureCount, numActions, kernel_size=1, stride=1, padding=0, bias=False)

        self.predictedExecutionTraceLinear = nn.Sequential(
            nn.Linear(self.pixelFeatureCount, executionTracePredictorSize),
            nn.Sigmoid()
        )

        self.numActions = numActions

    def forward(self, data):
        width = data['image'].shape[3]
        height = data['image'].shape[2]

        stamp = self.stampProjection(data['additionalFeature'])

        stampTiler = stamp.reshape([-1, self.branchStampEdgeSize, self.branchStampEdgeSize]).repeat([1, int(height / self.branchStampEdgeSize) + 1, int(width / self.branchStampEdgeSize) + 1])
        stampLayer = stampTiler[:, :height, :width].reshape([-1, 1, height, width])

        # Replace the saturation layer on the image with the stamp data layer
        merged = torch.cat([stampLayer, data['image']], dim=1)

        # print("Forward", merged.shape)
        output = self.mainModel(merged)
        # print("Output", output.shape)

        featureMap = output

        rewards = self.rewardConvolution(output)

        action_types = []
        action_xs = []
        action_ys = []
        if 'action_type' in data:
            action_types = data['action_type']
            action_xs = data['action_x']
            action_ys = data['action_y']
        else:
            action_indexes = rewards.reshape([-1, width * height * self.numActions]).argmax(1).data

            for index in action_indexes:
                action_type, action_x, action_y = BradNet.actionIndexToActionDetails(width, height, self.numActions, index)

                action_types.append(action_type)
                action_xs.append(action_x)
                action_ys.append(action_y)

        forwardFeatures = []
        for index, action_type, action_x, action_y in zip(range(len(action_types)), action_types, action_xs, action_ys):

            # temp fix
            action_x = min(action_x, width-1)
            action_y = min(action_y, height-1)

            featuresForTrace = featureMap[index, :, int(action_y), int(action_x)].unsqueeze(0)
            forwardFeatures.append(featuresForTrace)

        joinedFeatures = torch.cat(forwardFeatures, dim=0)

        predictedTraces = self.predictedExecutionTraceLinear(joinedFeatures)

        return rewards, predictedTraces


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.imageInputShape))).view(1, -1).size(1)


    @staticmethod
    def actionDetailsToActionIndex(width, height, numActions, action_type, action_x, action_y):
        return action_type + action_x * numActions + action_y * width * numActions


    @staticmethod
    def actionIndexToActionDetails(width, height, numActions, action_index):
        action_type = action_index % numActions

        action_x = int(int(action_index % (width * numActions)) / numActions)

        action_y = int(action_index / (width * numActions))

        return action_type, action_x, action_y

