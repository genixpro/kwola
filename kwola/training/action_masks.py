"""Rasterize configured browser action targets for model training."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from kwola.domain.actions import ActionChannel, ActionKind

from .geometry import Crop

ACTION_KINDS = tuple(ActionKind)


def action_masks(
    trace: Mapping[str, Any],
    size: tuple[int, int],
    channels: tuple[ActionChannel, ...] | None,
    crop: Crop | None = None,
) -> Tensor:
    width, height = size
    output_width = crop.width if crop is not None else width
    output_height = crop.height if crop is not None else height
    targets = trace.get("action_targets")
    if not isinstance(targets, list) or not targets:
        return torch.ones(len(channels or ACTION_KINDS), output_height, output_width)
    masks = torch.zeros(len(channels or ACTION_KINDS), output_height, output_width)
    supported = False
    for target in targets:
        supported |= _paint_target(
            masks,
            target,
            trace.get("viewport", [width, height]),
            channels,
            size,
            crop,
        )
    return masks if supported else torch.ones_like(masks)


def _paint_target(
    masks: Tensor,
    target: Mapping[str, Any],
    viewport: Sequence[int],
    channels: tuple[ActionChannel, ...] | None,
    size: tuple[int, int],
    crop: Crop | None,
) -> bool:
    width, height = size
    left, top, right, bottom = (int(value) for value in target["bounds"])
    x1 = min(width - 1, max(0, left * width // int(viewport[0])))
    y1 = min(height - 1, max(0, top * height // int(viewport[1])))
    x2 = min(width, max(x1 + 1, right * width // int(viewport[0])))
    y2 = min(height, max(y1 + 1, bottom * height // int(viewport[1])))
    if crop is not None:
        x1, x2 = max(x1, crop.left) - crop.left, min(x2, crop.right) - crop.left
        y1, y2 = max(y1, crop.top) - crop.top, min(y2, crop.bottom) - crop.top
    supported = False
    for index, channel in enumerate(channels or _generic_channels()):
        if _target_supports(target, channel):
            supported = True
            if x1 < x2 and y1 < y2:
                masks[index, y1:y2, x1:x2] = 1
    return supported


def _generic_channels() -> tuple[ActionChannel, ...]:
    return tuple(
        ActionChannel(
            kind.value,
            kind,
            1.0,
            text_strategy="letters" if kind is ActionKind.TYPE else None,
            direction="up" if kind is ActionKind.SCROLL else None,
        )
        for kind in ACTION_KINDS
    )


def _target_supports(target: Mapping[str, Any], channel: ActionChannel) -> bool:
    if channel.kind in {ActionKind.CLICK, ActionKind.DOUBLE_CLICK}:
        return bool(target.get("click"))
    if channel.kind is ActionKind.RIGHT_CLICK:
        return bool(target.get("right_click"))
    if channel.kind in {ActionKind.CLEAR, ActionKind.TYPE}:
        return bool(target.get("type"))
    return bool(target.get("scroll_up" if channel.direction == "up" else "scroll_down"))
