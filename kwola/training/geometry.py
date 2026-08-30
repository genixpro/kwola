"""Legacy-compatible screenshot scaling and deterministic training crops."""

import math
import random
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Crop:
    left: int
    top: int
    right: int
    bottom: int
    image_width: int
    image_height: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def aligned_size(width: int, height: int, ratio: float) -> tuple[int, int]:
    """Return legacy model dimensions, rounded upward to the convolution stride."""
    scaled_width = max(8, int(width * ratio))
    scaled_height = max(8, int(height * ratio))
    return _align(scaled_width), _align(scaled_height)


def process_screenshot(image: NDArray[np.uint8], ratio: float) -> NDArray[np.float32]:
    """Convert an RGB/BGR screenshot to the rounded grayscale model representation."""
    if image.ndim == 3:
        image = cast(NDArray[np.uint8], cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY))
    width, height = aligned_size(image.shape[1], image.shape[0], ratio)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return np.around(resized.astype(np.float32) / 255.0, decimals=2)


def centered_crop(
    center_x: float,
    center_y: float,
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
) -> Crop:
    """Match the historical crop clamping and integer truncation rules."""
    left = int(max(0, min(image_width - crop_width, center_x - crop_width / 2)))
    top = int(max(0, min(image_height - crop_height, center_y - crop_height / 2)))
    return Crop(left, top, left + crop_width, top + crop_height, image_width, image_height)


def action_crop(
    action_x: float,
    action_y: float,
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
    random_x: int,
    random_y: int,
    source: random.Random,
) -> Crop:
    return centered_crop(
        action_x + source.randint(-random_x, random_x),
        action_y + source.randint(-random_y, random_y),
        image_width,
        image_height,
        crop_width,
        crop_height,
    )


def random_crop(
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
    source: random.Random,
) -> Crop:
    minimum_x, maximum_x = _center_range(image_width)
    minimum_y, maximum_y = _center_range(image_height)
    return centered_crop(
        source.randint(minimum_x, maximum_x),
        source.randint(minimum_y, maximum_y),
        image_width,
        image_height,
        crop_width,
        crop_height,
    )


def scale_coordinate(value: int, source_size: int, target_size: int) -> int:
    return min(target_size - 1, max(0, int(value * target_size / source_size)))


def scale_bound(value: int, ratio: float, limit: int, *, lower: bool) -> int:
    adjusted = (value + 1) * ratio if lower else (value - 1) * ratio
    rounded = math.ceil(adjusted) if lower else math.floor(adjusted)
    return max(0, min(limit, int(rounded)))


def _align(value: int) -> int:
    return value if value % 8 == 0 else value + 8 - value % 8


def _center_range(size: int) -> tuple[int, int]:
    return (10, size - 10) if size >= 20 else (0, max(0, size - 1))
