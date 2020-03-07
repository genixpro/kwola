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

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
# Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs) if USE_CUDA else autograd.Variable(*args, **kwargs)



class BradNet(nn.Module):
    def __init__(self, additionalFeatureSize, numActions, executionTracePredictorSize):
        super(BradNet, self).__init__()

        self.branchStampEdgeSize = 20

        self.stampProjection = nn.Linear(additionalFeatureSize, self.branchStampEdgeSize*self.branchStampEdgeSize)

        self.mainModel = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=64)

        self.rewardConvolution = nn.Conv2d(64, numActions, 1, stride=1, padding=0, bias=False)

        self.predictedExecutionTraceLinear = nn.Sequential(
            nn.Linear(64, executionTracePredictorSize),
            nn.Sigmoid()
        )

        self.numActions = numActions

    def forward(self, data):
        width = data['image'].shape[3]
        height = data['image'].shape[2]

        stamp = self.stampProjection(data['additionalFeature'])

        stampTiler = stamp.reshape([self.branchStampEdgeSize, self.branchStampEdgeSize]).repeat([int(height / self.branchStampEdgeSize) + 1, int(width / self.branchStampEdgeSize) + 1])
        stampLayer = stampTiler[:height, :width].reshape([1, 1, height, width])

        # Replace the saturation layer on the image with the stamp data layer
        merged = torch.cat([stampLayer, data['image']], dim=1)

        output = self.mainModel(merged)

        featureMap = output['out']

        rewards = self.rewardConvolution(output['out'])

        action_index = rewards.reshape([-1, width * height * self.numActions]).argmax(1).data[0]
        action_type, action_x, action_y = self.actionIndexToActionDetails(width, height, action_index)

        featuresForTrace = featureMap[:, :, action_y, action_x]
        predictedTrace = self.predictedExecutionTraceLinear(featuresForTrace)

        return rewards, predictedTrace


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.imageInputShape))).view(1, -1).size(1)


    def actionDetailsToActionIndex(self, width, height, action_type, action_x, action_y):
        return action_type + action_x * self.numActions + action_y * width * self.numActions

    def actionIndexToActionDetails(self, width, height, action_index):
        action_type = action_index % self.numActions

        action_x = int(int(action_index % (width * self.numActions)) / self.numActions)
        action_y = int(action_index / (width * self.numActions))

        return action_type, action_x, action_y


    def predict(self, image, additionalFeatures, epsilon):
        width = image.shape[2]
        height = image.shape[1]

        if random.random() > epsilon:
            image = Variable(torch.FloatTensor(np.float32(image)).unsqueeze(0), volatile=True            )

            q_value, predictedTrace = self.forward({"image": image, "additionalFeature": Variable(torch.FloatTensor(additionalFeatures), volatile=True)})

            action_index = q_value.reshape([-1, width * height * self.numActions]).argmax(1).data[0]

            return self.actionIndexToActionDetails(width, height, action_index), predictedTrace
        else:

            action_type = random.randrange(self.numActions)
            action_x = random.randrange(width)
            action_y = random.randrange(height)

            return action_type, action_x, action_y

