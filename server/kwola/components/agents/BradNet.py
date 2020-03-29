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
    def __init__(self, agentConfiguration, additionalFeatureSize, numActions, executionTracePredictorSize, executionFeaturePredictorSize, cursorCount):
        super(BradNet, self).__init__()

        self.agentConfiguration = agentConfiguration

        self.agentConfiguration['additional_features_stamp_edge_size'] = agentConfiguration['additional_features_stamp_edge_size']

        self.stampProjection = nn.Linear(
            in_features=additionalFeatureSize,
            out_features=self.agentConfiguration['additional_features_stamp_edge_size'] * self.agentConfiguration['additional_features_stamp_edge_size']
        )

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
                out_channels=self.agentConfiguration['pixel_features'],
                kernel_size=self.agentConfiguration['layer_4_kernel_size'],
                stride=self.agentConfiguration['layer_4_stride'],
                dilation=self.agentConfiguration['layer_4_dilation'],
                padding=self.agentConfiguration['layer_4_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['pixel_features']),

            torch.nn.Upsample(scale_factor=8)
        )

        self.stateValueConvolution = nn.Sequential(
            nn.Conv2d(
                in_channels=self.agentConfiguration['pixel_features'],
                out_channels=self.agentConfiguration['layer_5_num_kernels'],
                kernel_size=self.agentConfiguration['layer_5_kernel_size'],
                stride=self.agentConfiguration['layer_5_stride'],
                dilation=self.agentConfiguration['layer_5_dilation'],
                padding=self.agentConfiguration['layer_5_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['pixel_features']),
            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_5_num_kernels'],
                out_channels=1,
                kernel_size=self.agentConfiguration['state_value_convolution_kernel_size'],
                stride=self.agentConfiguration['state_value_convolution_stride'],
                padding=self.agentConfiguration['state_value_convolution_padding'],
                bias=False
            )
        )

        self.presentRewardConvolution = nn.Sequential(
            nn.Conv2d(
                in_channels=self.agentConfiguration['pixel_features'],
                out_channels=self.agentConfiguration['layer_5_num_kernels'],
                kernel_size=self.agentConfiguration['layer_5_kernel_size'],
                stride=self.agentConfiguration['layer_5_stride'],
                dilation=self.agentConfiguration['layer_5_dilation'],
                padding=self.agentConfiguration['layer_5_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['pixel_features']),
            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.agentConfiguration['present_reward_convolution_kernel_size'],
                stride=self.agentConfiguration['present_reward_convolution_stride'],
                padding=self.agentConfiguration['present_reward_convolution_padding'],
                bias=False
            )
        )

        self.discountedFutureRewardConvolution = nn.Sequential(
            nn.Conv2d(
                in_channels=self.agentConfiguration['pixel_features'],
                out_channels=self.agentConfiguration['layer_5_num_kernels'],
                kernel_size=self.agentConfiguration['layer_5_kernel_size'],
                stride=self.agentConfiguration['layer_5_stride'],
                dilation=self.agentConfiguration['layer_5_dilation'],
                padding=self.agentConfiguration['layer_5_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['pixel_features']),
            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.agentConfiguration['discounted_future_reward_convolution_kernel_size'],
                stride=self.agentConfiguration['discounted_future_reward_convolution_stride'],
                padding=self.agentConfiguration['discounted_future_reward_convolution_padding'],
                bias=False
            )
        )

        self.actorConvolution = nn.Sequential(
            nn.Conv2d(
                in_channels=self.agentConfiguration['pixel_features'],
                out_channels=self.agentConfiguration['layer_5_num_kernels'],
                kernel_size=self.agentConfiguration['layer_5_kernel_size'],
                stride=self.agentConfiguration['layer_5_stride'],
                dilation=self.agentConfiguration['layer_5_dilation'],
                padding=self.agentConfiguration['layer_5_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['pixel_features']),
            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.agentConfiguration['discounted_future_reward_convolution_kernel_size'],
                stride=self.agentConfiguration['discounted_future_reward_convolution_stride'],
                padding=self.agentConfiguration['discounted_future_reward_convolution_padding'],
                bias=False
            )
        )

        self.advantageConvolution = nn.Sequential(
            nn.Conv2d(
                in_channels=self.agentConfiguration['pixel_features'],
                out_channels=self.agentConfiguration['layer_5_num_kernels'],
                kernel_size=self.agentConfiguration['layer_5_kernel_size'],
                stride=self.agentConfiguration['layer_5_stride'],
                dilation=self.agentConfiguration['layer_5_dilation'],
                padding=self.agentConfiguration['layer_5_padding']
            ),
            nn.ELU(),
            nn.BatchNorm2d(num_features=self.agentConfiguration['pixel_features']),
            nn.Conv2d(
                in_channels=self.agentConfiguration['layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.agentConfiguration['discounted_future_reward_convolution_kernel_size'],
                stride=self.agentConfiguration['discounted_future_reward_convolution_stride'],
                padding=self.agentConfiguration['discounted_future_reward_convolution_padding'],
                bias=False
            )
        )

        if self.agentConfiguration['enable_trace_prediction_loss']:
            self.predictedExecutionTraceLinear = nn.Sequential(
                nn.Linear(
                    in_features=self.agentConfiguration['pixel_features'],
                    out_features=executionTracePredictorSize
                ),
                nn.ELU()
            )

        if self.agentConfiguration['enable_execution_feature_prediction_loss']:
            self.predictedExecutionFeaturesLinear = nn.Sequential(
                nn.Linear(
                    in_features=self.agentConfiguration['pixel_features'],
                    out_features=executionFeaturePredictorSize
                ),
                nn.Sigmoid()
            )

        if self.agentConfiguration['enable_cursor_prediction_loss']:
            self.predictedCursorLinear = nn.Sequential(
                nn.Linear(
                    in_features=self.agentConfiguration['pixel_features'],
                    out_features=cursorCount
                ),
                nn.Sigmoid()
            )
        
        self.actionSoftmax = nn.Softmax(dim=1)

        self.numActions = numActions

    def forward(self, data):
        width = data['image'].shape[3]
        height = data['image'].shape[2]

        stamp = self.stampProjection(data['additionalFeature']).reshape([-1, self.agentConfiguration['additional_features_stamp_edge_size'], self.agentConfiguration['additional_features_stamp_edge_size']])

        stampTiler = stamp.repeat([1, int(height / self.agentConfiguration['additional_features_stamp_edge_size']) + 1, int(width / self.agentConfiguration['additional_features_stamp_edge_size']) + 1])
        stampLayer = stampTiler[:, :height, :width].reshape([-1, 1, height, width])

        # Append the stamp layer along side the main image brightness layer
        merged = torch.cat([stampLayer, data['image']], dim=1)

        pixelFeatureMap = self.mainModel(merged)

        outputDict = {}

        if data['computeRewards']:
            presentRewards = self.presentRewardConvolution(pixelFeatureMap) * data['pixelActionMaps'] + (1.0 - data['pixelActionMaps']) * self.agentConfiguration['reward_impossible_action']
            discountFutureRewards = self.discountedFutureRewardConvolution(pixelFeatureMap) * data['pixelActionMaps'] + (1.0 - data['pixelActionMaps']) * self.agentConfiguration['reward_impossible_action']

            totalReward = (presentRewards + discountFutureRewards)

            outputDict['presentRewards'] = presentRewards
            outputDict['discountFutureRewards'] = discountFutureRewards

        if data["outputStamp"]:
            outputDict["stamp"] = stamp.detach()

        if data["computeActionProbabilities"]:
            actorLogProbs = self.actorConvolution(pixelFeatureMap)

            actorActionProbs = torch.exp(actorLogProbs) * data['pixelActionMaps'] / torch.sum((torch.exp(actorLogProbs) * data['pixelActionMaps']).reshape(shape=[-1, width * height * self.numActions]), dim=1)
            actorActionProbs = actorActionProbs.reshape([-1, self.numActions, height, width])

            outputDict["actionProbabilities"] = actorActionProbs

        if data['computeStateValues']:
            stateValueMap = self.stateValueConvolution(pixelFeatureMap)

            for actionIndex in range(self.numActions):
                stateValueMap = stateValueMap * data['pixelActionMaps'][:, actionIndex]
            
            flatStateValueMap = stateValueMap.reshape([-1, width * height * self.numActions])

            averageStateValues = torch.sum(flatStateValueMap, dim=1) / (torch.sum(flatStateValueMap != 0, dim=1))

            outputDict['stateValues'] = averageStateValues

        if data['computeAdvantageValues']:
            advantageValues = self.advantageConvolution(pixelFeatureMap)
            outputDict['advantage'] = advantageValues

        if data['computeExtras']:
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
                featuresForAuxillaryLosses = pixelFeatureMap[sampleIndex, :, action_y, action_x].unsqueeze(0)
                forwardFeaturesForAuxillaryLosses.append(featuresForAuxillaryLosses)

            joinedFeatures = torch.cat(forwardFeaturesForAuxillaryLosses, dim=0)

            if self.agentConfiguration['enable_trace_prediction_loss']:
                outputDict['predictedTraces'] = self.predictedExecutionTraceLinear(joinedFeatures)
            if self.agentConfiguration['enable_execution_feature_prediction_loss']:
                outputDict['predictedExecutionFeatures'] = self.predictedExecutionFeaturesLinear(joinedFeatures)
            if self.agentConfiguration['enable_cursor_prediction_loss']:
                outputDict['predictedCursor'] = self.predictedCursorLinear(joinedFeatures)
            if self.agentConfiguration['enable_homogenization_loss']:
                outputDict['pixelFeatureMap'] = pixelFeatureMap

        return outputDict


    def feature_size(self):
        return self.features(torch.zeros(1, *self.imageInputShape)).view(1, -1).size(1)

