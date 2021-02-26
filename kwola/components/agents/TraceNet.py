#
#     Kwola is an AI algorithm that learns how to use other programs
#     automatically so that it can find bugs in them.
#
#     Copyright (C) 2020 Kwola Software Testing Inc.
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as
#     published by the Free Software Foundation, either version 3 of the
#     License, or (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


import torch
import torch.cuda
import torch.autograd.profiler as profiler

class TraceNet(torch.nn.Module):
    def __init__(self, config, numActions, executionFeaturePredictorSize, cursorCount):
        super(TraceNet, self).__init__()

        self.config = config

        self.stampSize = self.config['neural_network_additional_features_stamp_edge_size'] * \
                         self.config['neural_network_additional_features_stamp_edge_size'] * \
                         self.config['neural_network_additional_features_stamp_depth_size']

        self.timeEncodingSize = 1
        self.symbolEmbeddingPixelFeaturesSize = self.config['symbol_embedding_size'] + \
                                                self.config['neural_network_additional_features_stamp_depth_size'] + \
                                                self.config['neural_network_recent_actions_feature_size']

        self.recentSymbolEmbedding = torch.nn.EmbeddingBag(self.config['symbol_dictionary_size'], self.config['symbol_embedding_size'], mode="sum")
        self.coverageSymbolEmbedding = torch.nn.EmbeddingBag(self.config['symbol_dictionary_size'], self.config['symbol_embedding_size'], mode="sum")

        self.coverageAttentionKeyEmbedding = torch.nn.Embedding(self.config['symbol_dictionary_size'], self.config['symbol_embedding_size'])
        self.coverageAttentionValueEmbedding = torch.nn.Embedding(self.config['symbol_dictionary_size'], self.config['symbol_embedding_size'])
        self.coverageSymbolAttentionLayer = torch.nn.MultiheadAttention(self.config['symbol_embedding_size'], 2)

        self.queryProjection = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.config['neural_network_pixel_features'],
                out_features=self.config['symbol_embedding_size']
            )
        )

        self.recentSymbolStampProjection = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.config['symbol_embedding_size'],
                out_features=self.stampSize - self.timeEncodingSize
            ),
            torch.nn.ELU()
        )

        self.recentActionVectorProjection = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.config['testing_recent_actions_vector_number_of_recent_traces'] * numActions,
                out_features=self.config['neural_network_recent_actions_feature_size']
            ),
            torch.nn.ELU()
        )

        self.mainModel = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1 + numActions,
                out_channels=self.config['neural_network_layer_1_num_kernels'],
                kernel_size=self.config['neural_network_layer_1_kernel_size'],
                stride=self.config['neural_network_layer_1_stride'],
                dilation=self.config['neural_network_layer_1_dilation'],
                padding=self.config['neural_network_layer_1_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_layer_1_num_kernels']),

            torch.nn.Conv2d(
                in_channels=self.config['neural_network_layer_1_num_kernels'],
                out_channels=self.config['neural_network_layer_2_num_kernels'],
                kernel_size=self.config['neural_network_layer_2_kernel_size'],
                stride=self.config['neural_network_layer_2_stride'],
                dilation=self.config['neural_network_layer_2_dilation'],
                padding=self.config['neural_network_layer_2_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_layer_2_num_kernels']),

            torch.nn.Conv2d(
                in_channels=self.config['neural_network_layer_2_num_kernels'],
                out_channels=self.config['neural_network_layer_3_num_kernels'],
                kernel_size=self.config['neural_network_layer_3_kernel_size'],
                stride=self.config['neural_network_layer_3_stride'],
                dilation=self.config['neural_network_layer_3_dilation'],
                padding=self.config['neural_network_layer_3_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_layer_3_num_kernels']),

            torch.nn.Conv2d(
                in_channels=self.config['neural_network_layer_3_num_kernels'],
                out_channels=self.config['neural_network_pixel_features'],
                kernel_size=self.config['neural_network_layer_4_kernel_size'],
                stride=self.config['neural_network_layer_4_stride'],
                dilation=self.config['neural_network_layer_4_dilation'],
                padding=self.config['neural_network_layer_4_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_pixel_features'])
        )

        self.stateValueLinear = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=int(self.stampSize + self.config['symbol_embedding_size']),
                out_features=self.config['neural_network_layer_5_num_kernels']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_features=self.config['neural_network_layer_5_num_kernels']),
            torch.nn.Linear(
                in_features=self.config['neural_network_layer_5_num_kernels'],
                out_features=1
            )
        )

        self.presentRewardConvolution = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_pixel_features'] + self.symbolEmbeddingPixelFeaturesSize,
                out_channels=self.config['neural_network_layer_5_num_kernels'],
                kernel_size=self.config['neural_network_layer_5_kernel_size'],
                stride=self.config['neural_network_layer_5_stride'],
                dilation=self.config['neural_network_layer_5_dilation'],
                padding=self.config['neural_network_layer_5_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_layer_5_num_kernels']),
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.config['neural_network_present_reward_convolution_kernel_size'],
                stride=self.config['neural_network_present_reward_convolution_stride'],
                padding=self.config['neural_network_present_reward_convolution_padding'],
                bias=False
            ),
            torch.nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        )

        self.discountedFutureRewardConvolution = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_pixel_features'] + self.symbolEmbeddingPixelFeaturesSize,
                out_channels=self.config['neural_network_layer_5_num_kernels'],
                kernel_size=self.config['neural_network_layer_5_kernel_size'],
                stride=self.config['neural_network_layer_5_stride'],
                dilation=self.config['neural_network_layer_5_dilation'],
                padding=self.config['neural_network_layer_5_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_layer_5_num_kernels']),
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.config['neural_network_discounted_future_reward_convolution_kernel_size'],
                stride=self.config['neural_network_discounted_future_reward_convolution_stride'],
                padding=self.config['neural_network_discounted_future_reward_convolution_padding'],
                bias=False
            ),
            torch.nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        )

        self.actorConvolution = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_pixel_features'] + self.symbolEmbeddingPixelFeaturesSize,
                out_channels=self.config['neural_network_layer_5_num_kernels'],
                kernel_size=self.config['neural_network_layer_5_kernel_size'],
                stride=self.config['neural_network_layer_5_stride'],
                dilation=self.config['neural_network_layer_5_dilation'],
                padding=self.config['neural_network_layer_5_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_layer_5_num_kernels']),
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.config['neural_network_actor_convolution_kernel_size'],
                stride=self.config['neural_network_actor_convolution_stride'],
                padding=self.config['neural_network_actor_convolution_padding'],
                bias=False
            ),
            torch.nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        )

        self.advantageConvolution = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_pixel_features'] + self.symbolEmbeddingPixelFeaturesSize,
                out_channels=self.config['neural_network_layer_5_num_kernels'],
                kernel_size=self.config['neural_network_layer_5_kernel_size'],
                stride=self.config['neural_network_layer_5_stride'],
                dilation=self.config['neural_network_layer_5_dilation'],
                padding=self.config['neural_network_layer_5_padding']
            ),
            torch.nn.ELU(),
            torch.nn.BatchNorm2d(num_features=self.config['neural_network_layer_5_num_kernels']),
            torch.nn.Conv2d(
                in_channels=self.config['neural_network_layer_5_num_kernels'],
                out_channels=numActions,
                kernel_size=self.config['neural_network_advantage_convolution_kernel_size'],
                stride=self.config['neural_network_advantage_convolution_stride'],
                padding=self.config['neural_network_advantage_convolution_padding'],
                bias=False
            ),
            torch.nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False)
        )

        self.pixelFeatureMapUpsampler = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        )

        if self.config['enable_trace_prediction_loss']:
            self.predictedExecutionTraceLinear = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.config['neural_network_pixel_features'] + self.symbolEmbeddingPixelFeaturesSize,
                    out_features=self.config['symbol_embedding_size']
                ),
                torch.nn.ELU()
            )

        if self.config['enable_execution_feature_prediction_loss']:
            self.predictedExecutionFeaturesLinear = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.config['neural_network_pixel_features'] + self.symbolEmbeddingPixelFeaturesSize,
                    out_features=executionFeaturePredictorSize
                ),
                torch.nn.Sigmoid()
            )

        if self.config['enable_cursor_prediction_loss']:
            self.predictedCursorLinear = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.config['neural_network_pixel_features'] + self.symbolEmbeddingPixelFeaturesSize,
                    out_features=cursorCount
                ),
                torch.nn.Sigmoid()
            )

        self.actionSoftmax = torch.nn.Softmax(dim=1)

        self.numActions = numActions

    def forward(self, data):
        batchSize = data['image'].shape[0]
        width = data['image'].shape[3]
        height = data['image'].shape[2]

        inputImage = torch.cat([
            data['image'],
            data['recentActionsImage']
        ], dim=1)

        with profiler.record_function("main_convolution"):
            pixelFeatureMap = self.mainModel(inputImage)

        featureMapHeight = pixelFeatureMap.shape[2]
        featureMapWidth = pixelFeatureMap.shape[3]

        with profiler.record_function("embeddings"):
            # Compute the embedding based on the symbols provided. symbols are usually traces of which lines of code got executed.
            recentSymbolEmbeddings = self.recentSymbolEmbedding(data['recentSymbolIndexes'], data['recentSymbolOffsets'], per_sample_weights=data['recentSymbolWeights'])
            recentSymbolEmbeddings = torch.nn.functional.normalize(recentSymbolEmbeddings, dim=1)

            coverageSymbolEmbeddings = self.coverageSymbolEmbedding(data['coverageSymbolIndexes'], data['coverageSymbolOffsets'], per_sample_weights=data['coverageSymbolWeights'])
            coverageSymbolEmbeddings = torch.nn.functional.normalize(coverageSymbolEmbeddings, dim=1)

            coverageAttentionKeys = self.coverageAttentionKeyEmbedding(data['coverageSymbolsSet'])
            coverageAttentionValues = self.coverageAttentionValueEmbedding(data['coverageSymbolsSet'])

            recentActionFeatures = self.recentActionVectorProjection(data['recentActionsVector'])

        with profiler.record_function("attention"):
            pixelFeatureTransposed = pixelFeatureMap[:, :, :, :].transpose(1, 3).transpose(0, 2).reshape([batchSize * featureMapHeight * featureMapWidth, self.config['neural_network_pixel_features']])

            queries = self.queryProjection(pixelFeatureTransposed)
            queries = queries.reshape([-1, batchSize, self.config['symbol_embedding_size']])

            coverageAttentionKeys = coverageAttentionKeys.reshape([-1, 1, self.config['symbol_embedding_size']]).repeat([1, batchSize, 1])
            coverageAttentionValues = coverageAttentionValues.reshape([-1, 1, self.config['symbol_embedding_size']]).repeat([1, batchSize, 1])

            attentionResults, attentionResultWeights = self.coverageSymbolAttentionLayer(queries,
                                                                                         coverageAttentionKeys,
                                                                                         coverageAttentionValues,
                                                                                         data['coverageSymbolsKeyMask']
                                                                                         )

            symbolEmbeddingFeatures = attentionResults.reshape([featureMapHeight, featureMapWidth, batchSize, self.config['symbol_embedding_size']])
            symbolEmbeddingFeatures = symbolEmbeddingFeatures.transpose(0, 2).transpose(1, 3)

        with profiler.record_function("feature_amalgamation"):
            # Concatenate the step number with the rest of the additional features
            additionalFeaturesWithStep = torch.cat([
                torch.log10(data['stepNumber'] + torch.ones_like(data['stepNumber'])).reshape([-1, 1]),
                self.recentSymbolStampProjection(recentSymbolEmbeddings)
            ], dim=1)

            # Append the stamp layer along side the pixel-by-pixel features
            stamp = additionalFeaturesWithStep.reshape([-1, self.config['neural_network_additional_features_stamp_depth_size'],
                                                        self.config['neural_network_additional_features_stamp_edge_size'],
                                                        self.config['neural_network_additional_features_stamp_edge_size']])

            featureMapHeight = pixelFeatureMap.shape[2]
            featureMapWidth = pixelFeatureMap.shape[3]
            stampTiler = stamp.repeat([1, 1, int(featureMapHeight / self.config['neural_network_additional_features_stamp_edge_size']) + 1, int(featureMapWidth / self.config['neural_network_additional_features_stamp_edge_size']) + 1])
            stampLayer = stampTiler[:, :, :featureMapHeight, :featureMapWidth].reshape([-1, self.config['neural_network_additional_features_stamp_depth_size'], featureMapHeight, featureMapWidth])

            recentActionFeaturesImage = recentActionFeatures.reshape([batchSize, self.config['neural_network_recent_actions_feature_size'], 1, 1])
            recentActionFeaturesImage = recentActionFeaturesImage.repeat([1, 1, featureMapHeight, featureMapWidth])
            mergedPixelFeatureMap = torch.cat([symbolEmbeddingFeatures, recentActionFeaturesImage, stampLayer, pixelFeatureMap], dim=1)

        outputDict = {}

        if data['computeRewards']:
            with profiler.record_function("rewards"):
                presentRewardPredictions = self.presentRewardConvolution(mergedPixelFeatureMap)
                discountFutureRewardPredictions = self.discountedFutureRewardConvolution(mergedPixelFeatureMap)

                presentRewards = presentRewardPredictions * data['pixelActionMaps'] + (1.0 - data['pixelActionMaps']) * self.config['reward_impossible_action']
                discountFutureRewards = discountFutureRewardPredictions * data['pixelActionMaps'] + (1.0 - data['pixelActionMaps']) * self.config['reward_impossible_action']

                totalReward = (presentRewards + discountFutureRewards)

                outputDict['presentRewards'] = presentRewards
                outputDict['discountFutureRewards'] = discountFutureRewards
        else:
            totalReward = None

        if data["outputStamp"]:
            outputDict["stamp"] = stamp.detach()

        if data['outputFutureSymbolEmbedding']:
            # Compute the embedding based on the symbols provided for the future execution trace
            decayingFutureSymbolEmbedding = self.recentSymbolEmbedding(data['decayingFutureSymbolIndexes'],
                                                                       data['decayingFutureSymbolOffsets'],
                                                                       per_sample_weights=data['decayingFutureSymbolWeights'])


            outputDict['decayingFutureSymbolEmbedding'] = decayingFutureSymbolEmbedding

        if data["computeActionProbabilities"]:
            with profiler.record_function("action_probabilities"):
                # We have to do this clamp here to preserve numerical stability and prevent calculations from going
                # out of bounds in the torch.exp command below.
                actorLogProbs = self.actorConvolution(mergedPixelFeatureMap).clamp(-30, 30)

                actorProbExp = torch.exp(actorLogProbs) * data['pixelActionMaps']
                actorProbSums = torch.sum(actorProbExp.reshape(shape=[-1, width * height * self.numActions]), dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
                actorProbSums = torch.max(torch.eq(actorProbSums, 0).type_as(actorProbSums) * 1.0, actorProbSums)
                actorActionProbs = torch.true_divide(actorProbExp, actorProbSums)
                actorActionProbs = actorActionProbs.reshape([-1, self.numActions, height, width])

                outputDict["actionProbabilities"] = actorActionProbs.type_as(mergedPixelFeatureMap)

        if data['computeStateValues']:
            with profiler.record_function("state_value"):
                flatFeatureMap = torch.cat([
                    additionalFeaturesWithStep.reshape(shape=[batchSize, -1]),
                    coverageSymbolEmbeddings
                ], dim=1)

                stateValuePredictions = self.stateValueLinear(flatFeatureMap)

                outputDict['stateValues'] = stateValuePredictions

        if data['computeAdvantageValues']:
            with profiler.record_function("advantage"):
                advantageValues = self.advantageConvolution(mergedPixelFeatureMap) * data['pixelActionMaps'] + (1.0 - data['pixelActionMaps']) * self.config['reward_impossible_action']
                outputDict['advantage'] = advantageValues

        if data['computeExtras'] and (self.config['enable_trace_prediction_loss']
                                      or self.config['enable_execution_feature_prediction_loss']
                                      or self.config['enable_cursor_prediction_loss']):
            with profiler.record_function("extras"):
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
                    featuresForAuxillaryLosses = mergedPixelFeatureMap[sampleIndex, :, int(torch.floor_divide(action_y, 8)), int(torch.floor_divide(action_x, 8))].unsqueeze(0)
                    forwardFeaturesForAuxillaryLosses.append(featuresForAuxillaryLosses)

                joinedFeatures = torch.cat(forwardFeaturesForAuxillaryLosses, dim=0)

                if self.config['enable_trace_prediction_loss']:
                    outputDict['predictedTraces'] = self.predictedExecutionTraceLinear(joinedFeatures)
                if self.config['enable_execution_feature_prediction_loss']:
                    outputDict['predictedExecutionFeatures'] = self.predictedExecutionFeaturesLinear(joinedFeatures)
                if self.config['enable_cursor_prediction_loss']:
                    outputDict['predictedCursor'] = self.predictedCursorLinear(joinedFeatures)

        return outputDict

    def initialize(self):
        pass

    def computePresentRewardBounds(self):
        if hasattr(self, 'bestPossibleReward'):
            return self.worstPossibleReward, self.bestPossibleReward

        bestPossibleReward = 0.0
        worstPossibleReward = 0.0
        config = self.config

        bestPossibleReward += max(0.0, config['reward_action_success'])
        bestPossibleReward += max(0.0, config['reward_action_failure'])
        bestPossibleReward += max(0.0, config['reward_code_executed'])
        bestPossibleReward += max(0.0, config['reward_no_code_executed'])
        bestPossibleReward += max(0.0, config['reward_new_code_executed'])
        bestPossibleReward += max(0.0, config['reward_no_new_code_executed'])
        bestPossibleReward += max(0.0, config['reward_network_traffic'])
        bestPossibleReward += max(0.0, config['reward_no_network_traffic'])
        bestPossibleReward += max(0.0, config['reward_new_network_traffic'])
        bestPossibleReward += max(0.0, config['reward_no_new_network_traffic'])
        bestPossibleReward += max(0.0, config['reward_screenshot_changed'])
        bestPossibleReward += max(0.0, config['reward_no_screenshot_change'])
        bestPossibleReward += max(0.0, config['reward_new_screenshot'])
        bestPossibleReward += max(0.0, config['reward_no_new_screenshot'])
        bestPossibleReward += max(0.0, config['reward_url_changed'])
        bestPossibleReward += max(0.0, config['reward_no_url_change'])
        bestPossibleReward += max(0.0, config['reward_new_url'])
        bestPossibleReward += max(0.0, config['reward_no_new_url'])
        bestPossibleReward += max(0.0, config['reward_log_output'])
        bestPossibleReward += max(0.0, config['reward_no_log_output'])

        worstPossibleReward += min(0.0, config['reward_action_success'])
        worstPossibleReward += min(0.0, config['reward_action_failure'])
        worstPossibleReward += min(0.0, config['reward_code_executed'])
        worstPossibleReward += min(0.0, config['reward_no_code_executed'])
        worstPossibleReward += min(0.0, config['reward_new_code_executed'])
        worstPossibleReward += min(0.0, config['reward_no_new_code_executed'])
        worstPossibleReward += min(0.0, config['reward_network_traffic'])
        worstPossibleReward += min(0.0, config['reward_no_network_traffic'])
        worstPossibleReward += min(0.0, config['reward_new_network_traffic'])
        worstPossibleReward += min(0.0, config['reward_no_new_network_traffic'])
        worstPossibleReward += min(0.0, config['reward_screenshot_changed'])
        worstPossibleReward += min(0.0, config['reward_no_screenshot_change'])
        worstPossibleReward += min(0.0, config['reward_new_screenshot'])
        worstPossibleReward += min(0.0, config['reward_no_new_screenshot'])
        worstPossibleReward += min(0.0, config['reward_url_changed'])
        worstPossibleReward += min(0.0, config['reward_no_url_change'])
        worstPossibleReward += min(0.0, config['reward_new_url'])
        worstPossibleReward += min(0.0, config['reward_no_new_url'])
        worstPossibleReward += min(0.0, config['reward_log_output'])
        worstPossibleReward += min(0.0, config['reward_no_log_output'])

        self.bestPossibleReward = bestPossibleReward
        self.worstPossibleReward = worstPossibleReward

        return worstPossibleReward, bestPossibleReward

    def computeDiscountedFutureRewardBounds(self):
        oneFrameBounds = self.computePresentRewardBounds()

        low = oneFrameBounds[0] / (1.0 - self.config['reward_discount_rate'])
        high = oneFrameBounds[1] / (1.0 - self.config['reward_discount_rate'])

        return low, high

