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

        if whichGpu == "all":
            device_ids = [torch.device(f'cuda:{n}') for n in range(2)]
            output_device = device_ids[0]
        elif whichGpu is None:
            device_ids = None
            output_device = None
        else:
            device_ids = [torch.device(f'cuda:{whichGpu}')]
            output_device = device_ids[0]

        self.stampProjection = nn.Linear(
            in_features=additionalFeatureSize,
            out_features=self.agentConfiguration['additional_features_stamp_edge_size'] * self.agentConfiguration['additional_features_stamp_edge_size']
        )

        if whichGpu is not None:
            self.stampProjection = self.stampProjection.cuda(device=device_ids[0])
            self.stampProjectionParallel = nn.parallel.DistributedDataParallel(module=self.stampProjection, device_ids=device_ids, output_device=output_device)

        self.mainModel = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
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
        if whichGpu is not None:
            self.mainModel = self.mainModel.cuda(device=device_ids[0])
            self.mainModelParallel = nn.parallel.DistributedDataParallel(module=self.mainModel, device_ids=device_ids, output_device=output_device)

        self.presentRewardConvolution = nn.Conv2d(
            in_channels=self.agentConfiguration['pixel_features'],
            out_channels=numActions,
            kernel_size=self.agentConfiguration['present_reward_convolution_kernel_size'],
            stride=1,
            padding=0,
            bias=False
        )

        if whichGpu is not None:
            self.presentRewardConvolution = self.presentRewardConvolution.cuda(device=device_ids[0])
            self.presentRewardConvolutionParallel = nn.parallel.DistributedDataParallel(module=self.presentRewardConvolution, device_ids=device_ids, output_device=output_device)

        self.discountedFutureRewardConvolution = nn.Conv2d(
            in_channels=self.agentConfiguration['pixel_features'],
            out_channels=numActions,
            kernel_size=self.agentConfiguration['discounted_future_reward_convolution_kernel_size'],
            stride=1,
            padding=0,
            bias=False
        )

        if whichGpu is not None:
            self.discountedFutureRewardConvolution = self.discountedFutureRewardConvolution.cuda(device=device_ids[0])
            self.discountedFutureRewardConvolutionParallel = nn.parallel.DistributedDataParallel(module=self.discountedFutureRewardConvolution, device_ids=device_ids, output_device=output_device)

        self.predictedExecutionTraceLinear = nn.Sequential(
            nn.Linear(
                in_features=self.agentConfiguration['pixel_features'],
                out_features=executionTracePredictorSize
            ),
            nn.ELU()
        )

        if whichGpu is not None:
            self.predictedExecutionTraceLinear = self.predictedExecutionTraceLinear.cuda(device=device_ids[0])
            self.predictedExecutionTraceLinearParallel = nn.parallel.DistributedDataParallel(module=self.predictedExecutionTraceLinear, device_ids=device_ids, output_device=output_device)

        self.predictedExecutionFeaturesLinear = nn.Sequential(
            nn.Linear(
                in_features=self.agentConfiguration['pixel_features'],
                out_features=executionFeaturePredictorSize
            ),
            nn.Sigmoid()
        )
        if whichGpu is not None:
            self.predictedExecutionFeaturesLinear = self.predictedExecutionFeaturesLinear.cuda(device=device_ids[0])
            self.predictedExecutionFeaturesLinearParallel = nn.parallel.DistributedDataParallel(module=self.predictedExecutionFeaturesLinear, device_ids=device_ids, output_device=output_device)

        self.predictedCursorLinear = nn.Sequential(
            nn.Linear(
                in_features=self.agentConfiguration['pixel_features'],
                out_features=cursorCount
            ),
            nn.Sigmoid()
        )
        if whichGpu is not None:
            self.predictedCursorLinear = self.predictedCursorLinear.cuda(device=device_ids[0])
            self.predictedCursorLinearParallel = nn.parallel.DistributedDataParallel(module=self.predictedCursorLinear, device_ids=device_ids, output_device=output_device)

        self.actionSoftmax = nn.Softmax(dim=1)

        self.numActions = numActions

    @property
    def stampProjectionCurrent(self):
        if self.whichGpu is not None:
            return self.stampProjectionParallel
        else:
            return self.stampProjection

    @property
    def mainModelCurrent(self):
        if self.whichGpu is not None:
            return self.mainModelParallel
        else:
            return self.mainModel

    @property
    def presentRewardConvolutionCurrent(self):
        if self.whichGpu is not None:
            return self.presentRewardConvolutionParallel
        else:
            return self.presentRewardConvolution


    @property
    def discountedFutureRewardConvolutionCurrent(self):
        if self.whichGpu is not None:
            return self.discountedFutureRewardConvolutionParallel
        else:
            return self.discountedFutureRewardConvolution


    @property
    def predictedExecutionTraceLinearCurrent(self):
        if self.whichGpu is not None:
            return self.predictedExecutionTraceLinearParallel
        else:
            return self.predictedExecutionTraceLinear

    @property
    def predictedExecutionFeaturesLinearCurrent(self):
        if self.whichGpu is not None:
            return self.predictedExecutionFeaturesLinearParallel
        else:
            return self.predictedExecutionFeaturesLinear


    @property
    def predictedCursorLinearCurrent(self):
        if self.whichGpu is not None:
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

        if 'action_type' in data:
            action_types = data['action_type']
            action_xs = data['action_x']
            action_ys = data['action_y']
        else:
            action_types = []
            action_xs = []
            action_ys = []
            for sampleReward in totalReward:
                action_type = sampleReward.reshape([self.numActions, width * height]).max(dim=1)[0].argmax(0)
                action_types.append(action_type)

                action_y = sampleReward[action_type].max(dim=1)[0].argmax(0)
                action_ys.append(action_y)

                action_x = sampleReward[action_type, action_y].argmax(0)
                action_xs.append(action_x)

        forwardFeaturesForAuxillaryLosses = []
        for sampleIndex, action_type, action_x, action_y in zip(range(len(action_types)), action_types, action_xs, action_ys):
            # temp fix
            # action_x = min(action_x, width-1)
            # action_y = min(action_y, height-1)

            featuresForAuxillaryLosses = pixelFeatureMap[sampleIndex, :, action_y, action_x].unsqueeze(0)
            forwardFeaturesForAuxillaryLosses.append(featuresForAuxillaryLosses)

        joinedFeatures = torch.cat(forwardFeaturesForAuxillaryLosses, dim=0)

        predictedTraces = self.predictedExecutionTraceLinearCurrent(joinedFeatures)
        predictedExecutionFeatures = self.predictedExecutionFeaturesLinearCurrent(joinedFeatures)
        predictedCursor = self.predictedCursorLinearCurrent(joinedFeatures)
        actionProbabilities = self.actionSoftmax(totalReward.reshape([-1, self.numActions * height * width ])).reshape([-1, self.numActions, height, width])

        return presentRewards, discountFutureRewards, predictedTraces, predictedExecutionFeatures, predictedCursor, pixelFeatureMap, stamp, actionProbabilities


    def feature_size(self):
        return self.features(torch.zeros(1, *self.imageInputShape)).view(1, -1).size(1)

