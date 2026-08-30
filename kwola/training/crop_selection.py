"""Validity-preserving crop selection for bootstrapped next states."""

import random
from collections.abc import Mapping
from typing import Any

from kwola.domain.actions import ActionChannel

from .action_masks import action_masks
from .geometry import Crop, centered_crop, effective_crop, random_crop
from .sample_features import scaled_action


def valid_next_crop(
    trace: Mapping[str, Any],
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
    channels: tuple[ActionChannel, ...] | None,
    source: random.Random,
) -> Crop:
    """Sample next-state augmentation without hiding every valid action."""
    for _attempt in range(8):
        crop = effective_crop(
            random_crop(image_width, image_height, crop_width, crop_height, source)
        )
        if bool(action_masks(trace, (image_width, image_height), channels, crop=crop).any()):
            return crop

    action_x, action_y = scaled_action(trace, (image_width, image_height))
    fallback = effective_crop(
        centered_crop(
            action_x,
            action_y,
            image_width,
            image_height,
            crop_width,
            crop_height,
        )
    )
    if bool(action_masks(trace, (image_width, image_height), channels, crop=fallback).any()):
        return fallback
    raise ValueError("recorded next state has no valid action inside an action-centred crop")
