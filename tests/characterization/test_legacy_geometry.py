import random

import numpy as np
import torch

from kwola.training.geometry import action_crop, aligned_size, centered_crop, process_screenshot
from kwola.training.sample_features import reward_mask


def test_legacy_downscale_alignment_and_rounding() -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    image[:, 960:] = 255
    processed = process_screenshot(image, 0.3)

    assert aligned_size(1920, 1080, 0.3) == (576, 328)
    assert processed.shape == (328, 576)
    assert set(np.unique(processed)) == {0.0, 1.0}


def test_legacy_crop_equations_are_seeded_and_clamped() -> None:
    assert centered_crop(10, 10, 800, 600, 320, 320) == centered_crop(160, 160, 800, 600, 320, 320)
    crop = action_crop(400, 300, 800, 600, 320, 320, 100, 75, random.Random(9))

    assert (crop.left, crop.top, crop.right, crop.bottom) == (258, 160, 578, 480)
    oversized = centered_crop(100, 100, 300, 200, 448, 448)
    assert (oversized.left, oversized.top, oversized.right, oversized.bottom) == (0, 0, 448, 448)


def test_reward_mask_floods_equal_pixels_but_stays_inside_action_target() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 0.5
    trace = {
        "action": {"kind": "click", "x": 16, "y": 16},
        "viewport": [32, 32],
        "action_targets": [{"bounds": [8, 8, 24, 24]}],
    }

    mask = reward_mask(trace, (32, 32), processed_image=image)

    assert mask.dtype == torch.float32
    assert mask[16, 16] == 1
    assert mask[0, 0] == 0
    assert int(mask.sum()) > 25
