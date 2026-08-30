"""Compact exact inference inputs and reward maps for sampled debug sessions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kwola.domain.actions import ActionChannel

from .tracenet import TraceNetRequest


@dataclass(frozen=True, slots=True)
class InferenceDiagnostics:
    channel_names: tuple[str, ...]
    present_rewards: NDArray[np.float32] | None
    future_rewards: NDArray[np.float32] | None
    action_masks: NDArray[np.float32]
    recent_actions_image: NDArray[np.float32]
    recent_actions_vector: NDArray[np.float32]
    stamp: NDArray[np.float32] | None
    checkpoint_generation: int | None
    predicted_channel: str | None
    predicted_x: int | None
    predicted_y: int | None
    predicted_value: float | None
    coverage_symbol_count: int
    recent_symbol_count: int


def capture_inference_diagnostics(
    request: TraceNetRequest,
    output: dict[str, Tensor] | None,
    channels: tuple[ActionChannel, ...],
    *,
    checkpoint_generation: int | None,
    predicted: tuple[int, int, int, float] | None,
    map_downscale: int,
) -> InferenceDiagnostics:
    present = _optional_map(output, "presentRewards", map_downscale)
    future = _optional_map(output, "discountFutureRewards", map_downscale)
    stamp = None
    if output is not None and "stamp" in output:
        stamp = _array(output["stamp"][0])
    predicted_channel = None
    predicted_x = None
    predicted_y = None
    predicted_value = None
    if predicted is not None:
        channel, predicted_x, predicted_y, predicted_value = predicted
        predicted_channel = channels[channel].name
    backbone = request.backbone
    return InferenceDiagnostics(
        channel_names=tuple(channel.name for channel in channels),
        present_rewards=present,
        future_rewards=future,
        action_masks=_pool(request.pixel_action_maps[0], map_downscale, maximum=True),
        recent_actions_image=_pool(backbone.recent_actions_image[0], map_downscale, maximum=True),
        recent_actions_vector=_array(backbone.recent_actions_vector[0]),
        stamp=stamp,
        checkpoint_generation=checkpoint_generation,
        predicted_channel=predicted_channel,
        predicted_x=predicted_x,
        predicted_y=predicted_y,
        predicted_value=predicted_value,
        coverage_symbol_count=int(backbone.coverage_symbol_indexes.numel()),
        recent_symbol_count=int(backbone.recent_symbol_indexes.numel()),
    )


def _optional_map(
    output: dict[str, Tensor] | None, name: str, downscale: int
) -> NDArray[np.float32] | None:
    if output is None or name not in output:
        return None
    return _pool(output[name][0], downscale, maximum=False)


def _pool(tensor: Tensor, downscale: int, *, maximum: bool) -> NDArray[np.float32]:
    value = tensor.detach().float().unsqueeze(0)
    if downscale > 1:
        function = torch.nn.functional.max_pool2d if maximum else torch.nn.functional.avg_pool2d
        value = function(value, kernel_size=downscale, stride=downscale, ceil_mode=True)
    return _array(value[0])


def _array(tensor: Tensor) -> NDArray[np.float32]:
    return np.asarray(tensor.detach().cpu().numpy(), dtype=np.float32)
