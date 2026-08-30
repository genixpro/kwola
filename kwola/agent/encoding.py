"""Observation encoding and action-space masks for TraceNet inference."""

from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kwola.config.models import ModelConfig
from kwola.domain.actions import ActionChannel, ActionKind, ActionMap, ActionTarget
from kwola.domain.observations import Observation
from kwola.training.geometry import process_screenshot

from .model_backbone import BackboneInput
from .tracenet import TraceNetRequest

ACTION_KINDS = tuple(ActionKind)


class ObservationEncoder:
    def __init__(
        self,
        config: ModelConfig,
        edge: int | None,
        impossible_reward: float,
        channels: tuple[ActionChannel, ...] | None = None,
    ) -> None:
        self._config = config
        self._edge = edge
        self._impossible_reward = impossible_reward
        self._channels = channels

    def encode(self, observation: Observation, device: torch.device) -> TraceNetRequest:
        image = self._image(observation).to(device)
        height, width = image.shape[-2:]
        masks = action_masks(observation.action_map, (width, height), self._channels)
        masks = masks.unsqueeze(0).to(device)
        symbols = _symbols(observation, self._config.symbol_dictionary_size, device)
        offsets = torch.tensor([0], dtype=torch.long, device=device)
        weights = torch.ones(len(symbols), dtype=torch.float32, device=device)
        action_count = len(self._channels or ACTION_KINDS)
        recent_images = torch.zeros(1, action_count, height, width, device=device)
        recent_vector = torch.zeros(
            1, self._config.recent_action_history * action_count, device=device
        )
        backbone = BackboneInput(
            image,
            recent_images,
            recent_vector,
            symbols,
            offsets,
            weights,
            symbols,
            offsets,
            weights,
            symbols,
            torch.zeros(1, len(symbols), dtype=torch.bool, device=device),
            torch.tensor([1.0], device=device),
        )
        return TraceNetRequest(backbone, masks, self._impossible_reward)

    def _image(self, observation: Observation) -> Tensor:
        encoded = np.frombuffer(observation.screenshot, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError("observation contains an invalid screenshot")
        decoded_uint8 = cast(NDArray[np.uint8], decoded)
        resized: NDArray[np.float32]
        if self._edge is None:
            resized = process_screenshot(decoded_uint8, self._config.image_downscale_ratio)
        else:
            square = cv2.resize(
                decoded_uint8, (self._edge, self._edge), interpolation=cv2.INTER_AREA
            )
            resized = cast(NDArray[np.float32], square.astype(np.float32) / 255.0)
        return torch.from_numpy(resized).reshape(1, 1, *resized.shape)


def action_masks(
    action_map: ActionMap,
    size: int | tuple[int, int],
    channels: tuple[ActionChannel, ...] | None = None,
) -> Tensor:
    selected = channels or _generic_channels()
    width, height = (size, size) if isinstance(size, int) else size
    masks = torch.zeros(len(selected), height, width)
    for target in action_map.targets:
        _paint(masks, target, action_map, width, height, selected)
    return masks if bool(masks.any()) else torch.ones_like(masks)


def _paint(
    masks: Tensor,
    target: ActionTarget,
    action_map: ActionMap,
    width: int,
    height: int,
    channels: tuple[ActionChannel, ...],
) -> None:
    x1 = min(width - 1, max(0, target.left * width // action_map.viewport_width))
    y1 = min(height - 1, max(0, target.top * height // action_map.viewport_height))
    x2 = min(width, max(x1 + 1, target.right * width // action_map.viewport_width))
    y2 = min(height, max(y1 + 1, target.bottom * height // action_map.viewport_height))
    for index, channel in enumerate(channels):
        enabled = _enabled(channel, target)
        if enabled:
            masks[index, y1:y2, x1:x2] = 1


def _enabled(channel: ActionChannel, target: ActionTarget) -> bool:
    if channel.kind in {ActionKind.CLICK, ActionKind.DOUBLE_CLICK}:
        return target.can_click
    if channel.kind is ActionKind.RIGHT_CLICK:
        return target.can_right_click
    if channel.kind in {ActionKind.CLEAR, ActionKind.TYPE}:
        return target.can_type
    if channel.direction == "up":
        return target.can_scroll_up
    if channel.direction == "down":
        return target.can_scroll_down
    return False


def _generic_channels() -> tuple[ActionChannel, ...]:
    channels = []
    for kind in ACTION_KINDS:
        channels.append(
            ActionChannel(
                kind.value,
                kind,
                1.0,
                text_strategy="letters" if kind is ActionKind.TYPE else None,
                direction="up" if kind is ActionKind.SCROLL else None,
            )
        )
    return tuple(channels)


def _symbols(observation: Observation, size: int, device: torch.device) -> Tensor:
    raw = (*observation.branch_symbols, *observation.network_symbols)
    values = sorted({int(value) % size for value in raw} or {0})
    return torch.tensor(values, dtype=torch.long, device=device)
