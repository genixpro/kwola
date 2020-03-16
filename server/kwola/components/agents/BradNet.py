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

class BradNet(nn.Module):
    def __init__(self, agentConfiguration, additionalFeatureSize, numActions, executionTracePredictorSize, executionFeaturePredictorSize, cursorCount, whichGpu):
        super(BradNet, self).__init__()

        self.agentConfiguration = agentConfiguration

        self.agentConfiguration['additional_features_stamp_edge_size'] = agentConfiguration['additional_features_stamp_edge_size']

        self.whichGpu = whichGpu

        self.stampProjection = nn.Linear(
            in_features=additionalFeatureSize,
            out_features=self.agentConfiguration['additional_features_stamp_edge_size'] * self.agentConfiguration['additional_features_stamp_edge_size']
        )
        self.stampProjectionParallel = nn.DataParallel(module=self.stampProjection)

        self.mainModel = nn.Sequential(
            nn.Conv2d(
                in_channels=3, 
                out_channels=self.agentConfiguration['layer_1_num_kernels'],
                kernel_size=self.agentConfiguration['layer_1_kernel_size'], 
                stride=self.agentConfiguration['layer_1_stride'], 
                dilation=self.agentConfiguration['layer_1_dilation'], 
                padding=self.agentConfiguration['layer_1_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['layer_1_num_kernels']),

            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_1_num_kernels'],
                out_channels=self.agentConfiguration['layer_2_num_kernels'],
                kernel_size=self.agentConfiguration['layer_2_kernel_size'],
                stride=self.agentConfiguration['layer_2_stride'],
                dilation=self.agentConfiguration['layer_2_dilation'],
                padding=self.agentConfiguration['layer_2_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['layer_2_num_kernels']),

            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_2_num_kernels'],
                out_channels=self.agentConfiguration['layer_3_num_kernels'],
                kernel_size=self.agentConfiguration['layer_3_kernel_size'],
                stride=self.agentConfiguration['layer_3_stride'],
                dilation=self.agentConfiguration['layer_3_dilation'],
                padding=self.agentConfiguration['layer_3_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['layer_3_num_kernels']),

            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_3_num_kernels'],
                out_channels=self.agentConfiguration['layer_4_num_kernels'],
                kernel_size=self.agentConfiguration['layer_4_kernel_size'],
                stride=self.agentConfiguration['layer_4_stride'],
                dilation=self.agentConfiguration['layer_4_dilation'],
                padding=self.agentConfiguration['layer_4_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['layer_4_num_kernels']),

            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_4_num_kernels'],
                out_channels=self.agentConfiguration['pixel_features'],
                kernel_size=self.agentConfiguration['layer_5_kernel_size'],
                stride=self.agentConfiguration['layer_5_stride'],
                dilation=self.agentConfiguration['layer_5_dilation'],
                padding=self.agentConfiguration['layer_5_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['pixel_features']),

            torch.nn.Upsample(scale_factor=8)
        )
        self.mainModelParallel = nn.DataParallel(module=self.mainModel)

        self.presentRewardConvolution = nn.Conv2d(
            in_channels=self.agentConfiguration['pixel_features'],
            out_channels=numActions,
            kernel_size=self.agentConfiguration['present_reward_convolution_kernel_size'],
            stride=1,
            padding=0,
            bias=False
        )
        self.presentRewardConvolutionParallel = nn.DataParallel(module=self.presentRewardConvolution)

        self.discountedFutureRewardConvolution = nn.Conv2d(
            in_channels=self.agentConfiguration['pixel_features'],
            out_channels=numActions,
            kernel_size=self.agentConfiguration['discounted_future_reward_convolution_kernel_size'],
            stride=1,
            padding=0,
            bias=False
        )
        self.discountedFutureRewardConvolutionParallel = nn.DataParallel(module=self.discountedFutureRewardConvolution)

        self.predictedExecutionTraceLinear = nn.Sequential(
            nn.Linear(
                in_features=self.agentConfiguration['pixel_features'],
                out_features=executionTracePredictorSize
            ),
            nn.ELU()
        )
        self.predictedExecutionTraceLinearParallel = nn.DataParallel(module=self.predictedExecutionTraceLinear)

        self.predictedExecutionFeaturesLinear = nn.Sequential(
            nn.Linear(
                in_features=self.agentConfiguration['pixel_features'],
                out_features=executionFeaturePredictorSize
            ),
            nn.Sigmoid()
        )
        self.predictedExecutionFeaturesLinearParallel = nn.DataParallel(module=self.predictedExecutionFeaturesLinear)

        self.predictedCursorLinear = nn.Sequential(
            nn.Linear(
                in_features=self.agentConfiguration['pixel_features'],
                out_features=cursorCount
            ),
            nn.Sigmoid()
        )
        self.predictedCursorLinearParallel = nn.DataParallel(module=self.predictedCursorLinear)

        self.numActions = numActions

    @property
    def stampProjectionCurrent(self):
        if self.whichGpu == "all":
            return self.stampProjectionParallel
        else:
            return self.stampProjection

    @property
    def mainModelCurrent(self):
        if self.whichGpu == "all":
            return self.mainModelParallel
        else:
            return self.mainModel

    @property
    def presentRewardConvolutionCurrent(self):
        if self.whichGpu == "all":
            return self.presentRewardConvolutionParallel
        else:
            return self.presentRewardConvolution


    @property
    def discountedFutureRewardConvolutionCurrent(self):
        if self.whichGpu == "all":
            return self.discountedFutureRewardConvolutionParallel
        else:
            return self.discountedFutureRewardConvolution


    @property
    def predictedExecutionTraceLinearCurrent(self):
        if self.whichGpu == "all":
            return self.predictedExecutionTraceLinearParallel
        else:
            return self.predictedExecutionTraceLinear

    @property
    def predictedExecutionFeaturesLinearCurrent(self):
        if self.whichGpu == "all":
            return self.predictedExecutionFeaturesLinearParallel
        else:
            return self.predictedExecutionFeaturesLinear


    @property
    def predictedCursorLinearCurrent(self):
        if self.whichGpu == "all":
            return self.predictedCursorLinearParallel
        else:
            return self.predictedCursorLinear


    def forward(self, data):
        width = data['image'].shape[3]
        height = data['image'].shape[2]

        stamp = self.stampProjectionCurrent(data['additionalFeature']).reshape([-1, self.agentConfiguration['additional_features_stamp_edge_size'], self.agentConfiguration['additional_features_stamp_edge_size']])

        stampTiler = stamp.repeat([1, int(height / self.agentConfiguration['additional_features_stamp_edge_size']) + 1, int(width / self.agentConfiguration['additional_features_stamp_edge_size']) + 1])
        stampLayer = stampTiler[:, :height, :width].reshape([-1, 1, height, width])

        # Replace the saturation layer on the image with the stamp data layer
        merged = torch.cat([stampLayer, data['image']], dim=1)

        # print("Forward", merged.shape, flush=True)
        pixelFeatureMap = self.mainModelCurrent(merged)
        # print("Output", output.shape, flush=True)

        presentRewards = self.presentRewardConvolutionCurrent(pixelFeatureMap) * data['pixelActionMaps'] + (1.0 - data['pixelActionMaps']) * self.agentConfiguration['reward_impossible_action']
        discountFutureRewards = self.discountedFutureRewardConvolutionCurrent(pixelFeatureMap) * data['pixelActionMaps'] + (1.0 - data['pixelActionMaps']) * self.agentConfiguration['reward_impossible_action']
        totalReward = (presentRewards + discountFutureRewards)

        action_types = []
        action_xs = []
        action_ys = []
        if 'action_type' in data:
            action_types = data['action_type']
            action_xs = data['action_x']
            action_ys = data['action_y']
        else:
            action_indexes = totalReward.reshape([-1, width * height * self.numActions]).argmax(1).data

            for index in action_indexes:
                action_type, action_x, action_y = BradNet.actionIndexToActionDetails(width, height, self.numActions, index)

                action_types.append(action_type)
                action_xs.append(action_x)
                action_ys.append(action_y)

        forwardFeaturesForAuxillaryLosses = []
        for index, action_type, action_x, action_y in zip(range(len(action_types)), action_types, action_xs, action_ys):

            # temp fix
            action_x = min(action_x, width-1)
            action_y = min(action_y, height-1)

            featuresForAuxillaryLosses = pixelFeatureMap[index, :, int(action_y), int(action_x)].unsqueeze(0)
            forwardFeaturesForAuxillaryLosses.append(featuresForAuxillaryLosses)

        joinedFeatures = torch.cat(forwardFeaturesForAuxillaryLosses, dim=0)

        predictedTraces = self.predictedExecutionTraceLinearCurrent(joinedFeatures)
        predictedExecutionFeatures = self.predictedExecutionFeaturesLinearCurrent(joinedFeatures)
        predictedCursor = self.predictedCursorLinearCurrent(joinedFeatures)

        return presentRewards, discountFutureRewards, predictedTraces, predictedExecutionFeatures, predictedCursor, pixelFeatureMap, stamp


    def feature_size(self):
        return self.features(torch.zeros(1, *self.imageInputShape)).view(1, -1).size(1)


    @staticmethod
    def actionDetailsToActionIndex(width, height, numActions, action_type, action_x, action_y):
        return action_type * width * height \
               + action_y * width\
               + action_x


    @staticmethod
    def actionIndexToActionDetails(width, height, numActions, action_index):
        action_type = int(action_index / (width * height))

        index_within_type = int(action_index % (width * height))

        action_y = int(index_within_type / width)

        action_x = int(index_within_type % width)

        return action_type, action_x, action_y

