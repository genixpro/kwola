"""TraceNet prediction heads."""

from typing import cast

import torch
from torch import Tensor, nn

from kwola.config.models import ModelConfig


class TraceNetHeads(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        merged_features: int,
        num_actions: int,
        execution_feature_count: int,
        cursor_count: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_actions = num_actions
        stamp_size = config.additional_stamp_edge**2 * config.additional_stamp_depth
        layer_five = config.layers[4].kernels
        self.state_value = nn.Sequential(
            nn.Linear(stamp_size + config.symbol_embedding_size, layer_five),
            nn.ELU(),
            nn.BatchNorm1d(layer_five),
            nn.Linear(layer_five, 1),
        )
        self.present_reward = _map_head(config, merged_features, num_actions)
        self.discounted_reward = _map_head(config, merged_features, num_actions)
        self.actor = _map_head(config, merged_features, num_actions)
        self.advantage = _map_head(config, merged_features, num_actions)
        self.predicted_trace = _optional_linear(
            config.enable_trace_prediction, merged_features, config.symbol_embedding_size, nn.ELU()
        )
        self.predicted_execution = _optional_linear(
            config.enable_execution_feature_prediction,
            merged_features,
            execution_feature_count,
            nn.Sigmoid(),
        )
        self.predicted_cursor = _optional_linear(
            config.enable_cursor_prediction, merged_features, cursor_count, nn.Sigmoid()
        )

    def reward_maps(
        self, merged: Tensor, pixel_action_maps: Tensor, impossible: float
    ) -> tuple[Tensor, Tensor, Tensor]:
        present = _masked(self.present_reward(merged), pixel_action_maps, impossible)
        discounted = _masked(self.discounted_reward(merged), pixel_action_maps, impossible)
        return present, discounted, present + discounted

    def action_probabilities(self, merged: Tensor, masks: Tensor) -> Tensor:
        batch, _, height, width = masks.shape
        logits = self.actor(merged).clamp(-30, 30)
        exponentials = torch.exp(logits) * masks
        sums = exponentials.reshape(batch, height * width * self.num_actions).sum(dim=1)
        sums = sums[:, None, None, None]
        safe_sums = torch.maximum(torch.eq(sums, 0).type_as(sums), sums)
        return torch.true_divide(exponentials, safe_sums).reshape(
            batch, self.num_actions, height, width
        )


def _map_head(config: ModelConfig, inputs: int, outputs: int) -> nn.Sequential:
    layer = config.layers[4]
    return nn.Sequential(
        nn.Conv2d(
            inputs, layer.kernels, layer.kernel_size, layer.stride,
            layer.padding, layer.dilation,
        ),
        nn.ELU(),
        nn.BatchNorm2d(layer.kernels),
        nn.Conv2d(
            layer.kernels, outputs, config.prediction_head_kernel_size,
            config.prediction_head_stride, config.prediction_head_padding, bias=False,
        ),
        nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
    )


def _optional_linear(
    enabled: bool, inputs: int, outputs: int, activation: nn.Module
) -> nn.Module | None:
    return nn.Sequential(nn.Linear(inputs, outputs), activation) if enabled else None


def _masked(values: Tensor, masks: Tensor, impossible: float) -> Tensor:
    return cast(Tensor, values * masks + (1.0 - masks) * impossible)
