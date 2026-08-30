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
        self.visual_state_value = _VisualStateValue(merged_features, layer_five)

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

    def state_values(self, merged: Tensor, state_features: Tensor) -> Tensor:
        """Combine the historical symbolic critic with a visual residual critic."""
        symbolic = cast(Tensor, self.state_value(state_features))
        visual = cast(Tensor, self.visual_state_value(merged))
        return symbolic + visual

    def initialize_legacy_visual_state(self) -> None:
        """Keep old checkpoints behaviorally stable until their next optimizer step."""
        final = self.visual_state_value.projection[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)


class _VisualStateValue(nn.Module):
    """Resolution-independent visual critic retaining average and salient features."""

    def __init__(self, inputs: int, hidden: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(inputs * 2, hidden),
            nn.ELU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, merged: Tensor) -> Tensor:
        average = nn.functional.adaptive_avg_pool2d(merged, 1).flatten(1)
        maximum = nn.functional.adaptive_max_pool2d(merged, 1).flatten(1)
        return cast(Tensor, self.projection(torch.cat([average, maximum], dim=1)))


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
        nn.BatchNorm2d(layer.kernels),
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


def _optional_linear(
    enabled: bool, inputs: int, outputs: int, activation: nn.Module
) -> nn.Module | None:
    return nn.Sequential(nn.Linear(inputs, outputs), activation) if enabled else None


def _masked(values: Tensor, masks: Tensor, impossible: float) -> Tensor:
    return values * masks + (1.0 - masks) * impossible
