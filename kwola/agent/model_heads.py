"""TraceNet reward and optional auxiliary prediction heads."""

import torch
from torch import Tensor, nn

from kwola.config.models import ModelConfig

from .normalization import SpatialGroupNorm


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
        self.action_embedding = (
            nn.Embedding(num_actions, config.auxiliary_action_embedding_size)
            if config.enable_trace_prediction or config.enable_execution_feature_prediction
            else None
        )
        effect_inputs = merged_features + config.auxiliary_action_embedding_size
        self.present_reward = _map_head(config, merged_features, num_actions)
        self.discounted_reward = _map_head(config, merged_features, num_actions)
        self.predicted_trace = _optional_linear(
            config.enable_trace_prediction, effect_inputs, config.symbol_embedding_size
        )
        self.predicted_execution = _optional_linear(
            config.enable_execution_feature_prediction,
            effect_inputs,
            execution_feature_count,
        )
        self.predicted_cursor = _optional_linear(
            config.enable_cursor_prediction, merged_features, cursor_count
        )

    def reward_maps(self, merged: Tensor) -> tuple[Tensor, Tensor]:
        return self.present_reward(merged), self.discounted_reward(merged)

    @staticmethod
    def action_values(
        present: Tensor, discounted: Tensor, masks: Tensor, impossible: float
    ) -> Tensor:
        return _masked(present + discounted, masks, impossible)


def _map_head(config: ModelConfig, inputs: int, outputs: int) -> nn.Sequential:
    layer = config.layers[4]
    return nn.Sequential(
        nn.Conv2d(
            inputs,
            layer.kernels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
        ),
        nn.ELU(),
        SpatialGroupNorm(layer.kernels),
        nn.Conv2d(
            layer.kernels,
            outputs,
            config.prediction_head_kernel_size,
            config.prediction_head_stride,
            config.prediction_head_padding,
            bias=False,
        ),
        nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),
    )


def _optional_linear(enabled: bool, inputs: int, outputs: int) -> nn.Module | None:
    return nn.Linear(inputs, outputs) if enabled else None


def _masked(values: Tensor, masks: Tensor, impossible: float) -> Tensor:
    del impossible
    return values.masked_fill(masks <= 0, torch.finfo(values.dtype).min)
