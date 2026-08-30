"""Observation encoding and action-space masks for TraceNet inference."""

from collections.abc import Sequence
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from kwola.config.models import ModelConfig
from kwola.domain.actions import Action, ActionChannel, ActionKind, ActionMap, ActionTarget
from kwola.domain.observations import Observation
from kwola.training.geometry import process_screenshot

from .model_backbone import BackboneInput
from .spatial import coordinate_image
from .tracenet import TraceNetRequest

ACTION_KINDS = tuple(ActionKind)


class ObservationEncoder:
    def __init__(
        self,
        config: ModelConfig,
        edge: int | None,
        impossible_reward: float,
        channels: tuple[ActionChannel, ...] | None = None,
        recent_action_radius: int = 40,
        recent_action_decay: float = 0.8,
    ) -> None:
        self._config = config
        self._edge = edge
        self._impossible_reward = impossible_reward
        self._channels = channels
        self._recent_action_radius = recent_action_radius
        self._recent_action_decay = recent_action_decay

    def encode(
        self,
        observation: Observation,
        device: torch.device,
        *,
        recent_actions: Sequence[Action] = (),
        coverage_symbols: Sequence[int] | None = None,
        recent_symbol_history: Sequence[Sequence[int]] | None = None,
        step_number: int = 1,
    ) -> TraceNetRequest:
        image = self._image(observation).to(device)
        height, width = image.shape[-2:]
        masks, action_map_available = action_masks_with_availability(
            observation.action_map, (width, height), self._channels
        )
        masks = masks.unsqueeze(0).to(device)
        coverage = _symbols(
            coverage_symbols if coverage_symbols is not None else observation.branch_symbols,
            self._config.symbol_dictionary_size,
            device,
        )
        recent_indexes, recent_weights = _recent_symbols(
            recent_symbol_history
            if recent_symbol_history is not None
            else (observation.branch_symbols,),
            self._config.symbol_dictionary_size,
            device,
        )
        offsets = torch.tensor([0], dtype=torch.long, device=device)
        coverage_weights = torch.ones(len(coverage), dtype=torch.float32, device=device)
        action_count = len(self._channels or ACTION_KINDS)
        recent_images = torch.zeros(1, action_count, height, width, device=device)
        recent_vector = torch.zeros(
            1, self._config.recent_action_history * action_count, device=device
        )
        _recent_action_features(
            recent_images[0],
            recent_vector[0],
            recent_actions,
            observation,
            self._channels,
            self._config.recent_action_history,
            self._recent_action_radius,
            self._recent_action_decay,
        )
        backbone = BackboneInput(
            image,
            recent_images,
            masks,
            coordinate_image(width, height, device=device).unsqueeze(0),
            torch.full(
                (1, 1, height, width),
                float(action_map_available),
                dtype=image.dtype,
                device=device,
            ),
            recent_vector,
            recent_indexes,
            offsets,
            recent_weights,
            coverage,
            offsets,
            coverage_weights,
            coverage,
            torch.zeros(1, len(coverage), dtype=torch.bool, device=device),
            torch.tensor([float(step_number)], device=device),
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
    return action_masks_with_availability(action_map, size, channels)[0]


def action_masks_with_availability(
    action_map: ActionMap,
    size: int | tuple[int, int],
    channels: tuple[ActionChannel, ...] | None = None,
) -> tuple[Tensor, bool]:
    selected = channels or _generic_channels()
    width, height = (size, size) if isinstance(size, int) else size
    masks = torch.zeros(len(selected), height, width)
    for target in action_map.targets:
        _paint(masks, target, action_map, width, height, selected)
    available = bool(masks.any())
    return (masks if available else torch.ones_like(masks)), available


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


def _symbols(raw: Sequence[int], size: int, device: torch.device) -> Tensor:
    values = sorted({int(value) % size for value in raw} or {0})
    return torch.tensor(values, dtype=torch.long, device=device)


def _recent_symbols(
    history: Sequence[Sequence[int]], size: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    weighted: dict[int, float] = {}
    for age, symbols in enumerate(reversed(history)):
        weight = 0.9**age
        for symbol in symbols:
            mapped = int(symbol) % size
            weighted[mapped] = min(1.0, weighted.get(mapped, 0.0) + weight)
    if not weighted:
        weighted[0] = 1.0
    ordered = sorted(weighted.items())
    return (
        torch.tensor([symbol for symbol, _weight in ordered], dtype=torch.long, device=device),
        torch.tensor([weight for _symbol, weight in ordered], dtype=torch.float32, device=device),
    )


def _recent_action_features(
    image: Tensor,
    vector: Tensor,
    actions: Sequence[Action],
    observation: Observation,
    channels: tuple[ActionChannel, ...] | None,
    history: int,
    radius: int,
    decay: float,
) -> None:
    selected = channels or _generic_channels()
    height, width = image.shape[-2:]
    for position, action in enumerate(reversed(actions[-history:])):
        action_index = _action_index(action, selected)
        vector[position * len(selected) + action_index] = 1
        x = min(width - 1, max(0, action.x * width // observation.viewport.width))
        y = min(height - 1, max(0, action.y * height // observation.viewport.height))
        _paint_action_circle(
            image[action_index], x, y, min(radius, min(height, width)), decay**position
        )


def _action_index(action: Action, channels: tuple[ActionChannel, ...]) -> int:
    names = tuple(channel.name for channel in channels)
    if action.channel_name in names:
        return names.index(action.channel_name)
    for index, channel in enumerate(channels):
        if channel.kind is action.kind and channel.direction == action.direction:
            return index
    raise ValueError(f"action channel {action.channel_name!r} is not configured")


def _paint_action_circle(image: Tensor, x: int, y: int, radius: int, gain: float) -> None:
    left = max(0, x - radius + 1)
    right = min(image.shape[1], x + radius)
    top = max(0, y - radius + 1)
    bottom = min(image.shape[0], y + radius)
    if left >= right or top >= bottom:
        return
    yy, xx = torch.meshgrid(
        torch.arange(top - y, bottom - y, device=image.device),
        torch.arange(left - x, right - x, device=image.device),
        indexing="ij",
    )
    distance = torch.sqrt(xx.square() + yy.square())
    circle = torch.where(
        distance < radius,
        ((radius - distance) / radius * 0.7 + 0.3) * gain,
        torch.zeros_like(distance),
    )
    target = image[top:bottom, left:right]
    target.copy_(torch.minimum(torch.ones_like(target), target + circle))
