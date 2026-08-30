"""Recorded screenshot validation and crop preparation."""

import random
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kwola.domain.actions import ActionChannel

from .crop_selection import valid_next_crop
from .geometry import Crop, action_crop, centered_crop, effective_crop
from .image_cache import DecodedImageCache
from .sample_features import scaled_action


@dataclass(frozen=True, slots=True)
class PreparedSample:
    key: str
    trace: dict[str, Any]
    image: NDArray[np.float32]
    crop: Crop


def prepare_sample(
    run_dir: Path,
    images: DecodedImageCache,
    channels: tuple[ActionChannel, ...] | None,
    crop_size: tuple[int, int] | None,
    next_crop_size: tuple[int, int] | None,
    crop_random: tuple[int, int],
    source: random.Random,
    item: tuple[str, dict[str, Any]],
    edge: int | None,
    *,
    current: bool,
    full: bool = False,
) -> PreparedSample:
    key, trace = item
    path = run_dir / str(trace.get("screenshot_before", trace["screenshot"]))
    image = images.decode(path, edge)
    desired = crop_size if current else next_crop_size
    if edge is not None:
        desired = (edge, edge)
    desired = desired or (image.shape[1], image.shape[0])
    if full:
        crop = Crop(0, 0, image.shape[1], image.shape[0], image.shape[1], image.shape[0])
    elif current:
        x, y = scaled_action(trace, (image.shape[1], image.shape[0]))
        crop = action_crop(
            x,
            y,
            image.shape[1],
            image.shape[0],
            *desired,
            *crop_random,
            source,
        )
    elif edge is not None:
        crop = centered_crop(
            image.shape[1] / 2,
            image.shape[0] / 2,
            image.shape[1],
            image.shape[0],
            *desired,
        )
    else:
        crop = valid_next_crop(trace, image.shape[1], image.shape[0], *desired, channels, source)
    return PreparedSample(key, trace, image, effective_crop(crop))


def validate_screenshots(
    run_dir: Path, traces: Sequence[tuple[str, dict[str, Any]]], workers: int
) -> None:
    validate = partial(_validate_screenshot, run_dir)
    if workers > 0:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            tuple(executor.map(validate, (trace for _key, trace in traces)))
        return
    for _key, trace in traces:
        validate(trace)


def _validate_screenshot(run_dir: Path, trace: Mapping[str, Any]) -> None:
    path = run_dir / str(trace.get("screenshot_before", trace["screenshot"]))
    if not path.is_file() or not path.read_bytes():
        raise ValueError(f"missing or empty screenshot blob: {path}")
