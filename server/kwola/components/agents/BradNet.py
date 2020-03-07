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
    def __init__(self, branchFeatureSize, numActions):
        super(BradNet, self).__init__()

        self.branchStampEdgeSize = 10

        self.stampProjection = nn.Linear(branchFeatureSize, self.branchStampEdgeSize*self.branchStampEdgeSize)

        self.mainModel = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=64)

        self.rewardConvolution = nn.Conv2d(64, numActions, 1, stride=1, padding=0, bias=False)

        self.numActions = numActions

    def forward(self, data):
        outWidth = data['image'].shape[3]
        outHeight = data['image'].shape[2]

        stamp = self.stampProjection(data['branchFeature'])

        stampTiler = stamp.reshape([self.branchStampEdgeSize, self.branchStampEdgeSize]).repeat([int(outHeight / self.branchStampEdgeSize) + 1, int(outWidth / self.branchStampEdgeSize) + 1])
        stampLayer = stampTiler[:outHeight, :outWidth].reshape([1, 1, outHeight, outWidth])

        # Replace the saturation layer on the image with the stamp data layer
        merged = torch.cat([stampLayer, data['image']], dim=1)

        output = self.mainModel(merged)

        rewards = self.rewardConvolution(output['out'])

        return rewards


    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.imageInputShape))).view(1, -1).size(1)


    def actionDetailsToActionIndex(self, width, height, action_type, action_x, action_y):
        return action_type + action_x * self.numActions + action_y * width * self.numActions

    def actionIndexToActionDetails(self, width, height, action_index):
        action_type = action_index % self.numActions

        action_x = int(action_index % (width * self.numActions))
        action_y = int(action_index / (width * self.numActions))

        return action_type, action_x, action_y


    def predict(self, image, branchFeature, epsilon):
        width = image.shape[2]
        height = image.shape[1]

        if random.random() > epsilon:
            image = Variable(torch.FloatTensor(np.float32(image)).unsqueeze(0), volatile=True            )

            q_value = self.forward({"image": image, "branchFeature": Variable(torch.FloatTensor(branchFeature), volatile=True) })

            action_index = q_value.reshape([-1, width * height * self.numActions]).argmax(1).data[0]

            return self.actionIndexToActionDetails(width, height, action_index)
        else:

            action_type = random.randrange(self.numActions)
            action_x = random.randrange(width)
            action_y = random.randrange(height)

            return action_type, action_x, action_y

