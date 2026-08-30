"""Rasterize configured browser action targets for model training."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from kwola.domain.actions import ActionChannel, ActionKind

ACTION_KINDS = tuple(ActionKind)


def action_masks(
    trace: Mapping[str, Any], size: tuple[int, int], channels: tuple[ActionChannel, ...] | None
) -> Tensor:
    width, height = size
    targets = trace.get("action_targets")
    if not isinstance(targets, list) or not targets:
        return torch.ones(len(channels or ACTION_KINDS), height, width)
    masks = torch.zeros(len(channels or ACTION_KINDS), height, width)
    for target in targets:
        _paint_target(masks, target, trace.get("viewport", [width, height]), channels)
    return masks if bool(masks.any()) else torch.ones_like(masks)


def _paint_target(
    masks: Tensor,
    target: Mapping[str, Any],
    viewport: Sequence[int],
    channels: tuple[ActionChannel, ...] | None,
) -> None:
    height, width = masks.shape[-2:]
    left, top, right, bottom = (int(value) for value in target["bounds"])
    x1 = min(width - 1, max(0, left * width // int(viewport[0])))
    y1 = min(height - 1, max(0, top * height // int(viewport[1])))
    x2 = min(width, max(x1 + 1, right * width // int(viewport[0])))
    y2 = min(height, max(y1 + 1, bottom * height // int(viewport[1])))
    for index, channel in enumerate(channels or _generic_channels()):
        if _target_supports(target, channel):
            masks[index, y1:y2, x1:x2] = 1


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
