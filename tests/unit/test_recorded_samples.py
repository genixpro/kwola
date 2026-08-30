from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np
import torch

from kwola.storage import AtomicBlobStore, LmdbRunStore
from kwola.training.action_masks import action_masks
from kwola.training.geometry import Crop, process_screenshot
from kwola.training.image_cache import DecodedImageCache
from kwola.training.sample_features import (
    _flood_reward,
    _paint_action_circle,
    coverage_symbols,
    recent_symbols,
    step_symbol_features,
    symbol_features,
)
from kwola.training.samples import ACTION_KINDS, RecordedSampleAssembler


def test_recorded_samples_rebuild_from_trace_artifacts(tmp_path: Path) -> None:
    blobs = AtomicBlobStore(tmp_path / "blobs")
    screenshot = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    encoded_ok, encoded = cv2.imencode(".png", screenshot)
    assert encoded_ok
    path = blobs.write("screenshots", "trace.png", encoded.tobytes())
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        for index in range(2):
            store.put(
                "traces",
                f"testing-0-trace-{index:04d}",
                _trace(index, str(path.relative_to(tmp_path))),
            )
        assembler = RecordedSampleAssembler(
            tmp_path,
            store,
            symbol_dictionary_size=100,
            discount_rate=0.85,
            max_discounted_reward=10.0,
            cache_version=3,
            decoded_image_cache_size=4,
            freeze_records=True,
        )
        assert assembler.prepare_step("testing-0") == 2
        batch = assembler.assemble(
            batch_size=2,
            edge=64,
            device=torch.device("cpu"),
            impossible_reward=-10.0,
        )
        assert batch.request.backbone.image.shape == (2, 1, 64, 64)
        assert batch.request.pixel_action_maps.shape == (2, len(ACTION_KINDS), 64, 64)
        assert batch.next_state_valid.tolist() == [True, False]
        assert batch.present_rewards.tolist() == [0.5, 1.5]
        compact = RecordedSampleAssembler(
            tmp_path,
            store,
            symbol_dictionary_size=100,
            discount_rate=0.85,
            max_discounted_reward=10.0,
            cache_version=3,
            decoded_image_cache_size=4,
            compact_cpu_tensors=True,
            freeze_records=True,
        ).assemble(
            batch_size=2,
            edge=64,
            device=torch.device("cpu"),
            impossible_reward=-10.0,
        )
        assert compact.request.pixel_action_maps.dtype is torch.uint8
        torch.testing.assert_close(
            compact.request.pixel_action_maps.float(), batch.request.pixel_action_maps
        )
        assert store.get("sample_cache", "testing-0") == {
            "cache_version": 3,
            "payload": {"trace_ids": ["testing-0-trace-0000", "testing-0-trace-0001"]},
        }
        lean = RecordedSampleAssembler(
            tmp_path,
            store,
            symbol_dictionary_size=100,
            discount_rate=0.85,
            max_discounted_reward=10.0,
            cache_version=3,
            decoded_image_cache_size=4,
            freeze_records=True,
            enable_trace_prediction=False,
            enable_execution_feature_prediction=False,
            enable_cursor_prediction=False,
        ).assemble(
            batch_size=2,
            edge=64,
            device=torch.device("cpu"),
            impossible_reward=-10.0,
        )
        assert not lean.request.compute_auxiliary
        assert lean.request.future_symbol_indexes is None
        assert lean.execution_features is None
        assert lean.cursors is None
        path.unlink()
        cached = assembler.assemble(
            batch_size=2,
            edge=64,
            device=torch.device("cpu"),
            impossible_reward=-10.0,
        )
        assert cached.sample_ids == batch.sample_ids


def test_local_action_circle_matches_full_frame_calculation() -> None:
    expected = torch.rand(96, 128)
    actual = expected.clone()
    _legacy_paint_action_circle(expected, 3, 92, 40, 0.64)
    _paint_action_circle(actual, 3, 92, 40, 0.64)
    torch.testing.assert_close(actual, expected)


def test_in_place_flood_reward_matches_copy_per_seed_calculation() -> None:
    source = np.random.default_rng(7)
    image = source.integers(0, 8, size=(64, 96), dtype=np.uint8).astype(np.float32) / 100
    allowed = source.integers(0, 2, size=image.shape, dtype=np.uint8)
    expected = _legacy_flood_reward(image, allowed, 47, 31)
    actual = _flood_reward(image, allowed, 47, 31)
    np.testing.assert_array_equal(actual, expected)


def test_crop_local_action_masks_match_full_frame_slice() -> None:
    trace = _trace(0, "unused.png")
    trace["viewport"] = [1920, 1080]
    trace["action_targets"] = [
        {"bounds": [100, 80, 400, 300], "click": True},
        {"bounds": [1500, 700, 1800, 900], "type": True},
    ]
    crop = Crop(128, 64, 448, 384, 1920, 1080)

    full = action_masks(trace, (1920, 1080), None)
    local = action_masks(trace, (1920, 1080), None, crop=crop)

    torch.testing.assert_close(local, full[:, crop.top : crop.bottom, crop.left : crop.right])


def test_crop_local_action_masks_preserve_supported_target_fallback() -> None:
    crop = Crop(0, 0, 320, 320, 1920, 1080)
    outside = _trace(0, "unused.png")
    outside["viewport"] = [1920, 1080]
    outside["action_targets"] = [
        {"bounds": [1500, 700, 1800, 900], "click": True},
    ]
    unsupported = _trace(0, "unused.png")
    unsupported["viewport"] = [1920, 1080]
    unsupported["action_targets"] = [
        {"bounds": [1500, 700, 1800, 900]},
    ]

    assert not bool(action_masks(outside, (1920, 1080), None, crop=crop).any())
    assert bool(action_masks(unsupported, (1920, 1080), None, crop=crop).all())


def test_decoded_image_cache_matches_live_grayscale_encoding(tmp_path: Path) -> None:
    source = np.zeros((40, 64, 3), dtype=np.uint8)
    source[:, :, 0] = 250
    source[10:30, 20:50, 1] = 180
    encoded_ok, encoded = cv2.imencode(".png", source)
    assert encoded_ok
    path = tmp_path / "color.png"
    path.write_bytes(encoded.tobytes())
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    assert decoded is not None

    cache_dir = tmp_path / "cache"
    actual = DecodedImageCache(1, 0.3, cache_dir).decode(path, None)

    np.testing.assert_array_equal(actual, process_screenshot(decoded, 0.3))
    path.unlink()
    persisted = DecodedImageCache(1, 0.3, cache_dir).decode(path, None)
    np.testing.assert_array_equal(persisted, actual)


def test_combined_symbol_features_match_individual_builders() -> None:
    traces = [(f"trace-{index}", _trace(index, "unused.png")) for index in range(8)]
    current = traces[6][1]

    coverage, recent = symbol_features(current, traces, 100)

    assert coverage == coverage_symbols(current, traces, 100)
    assert recent == recent_symbols(current, traces, 100)

    indexed = step_symbol_features(traces, 100)
    for key, trace in traces:
        expected_coverage = coverage_symbols(trace, traces, 100)
        expected_recent = recent_symbols(trace, traces, 100)
        assert indexed[key][0] == expected_coverage
        assert [item[0] for item in indexed[key][1]] == [item[0] for item in expected_recent]
        np.testing.assert_allclose(
            [item[1] for item in indexed[key][1]],
            [item[1] for item in expected_recent],
            rtol=1e-12,
            atol=1e-12,
        )


def _legacy_paint_action_circle(
    image: torch.Tensor, x: int, y: int, radius: int, gain: float
) -> None:
    yy, xx = torch.meshgrid(
        torch.arange(image.shape[0]), torch.arange(image.shape[1]), indexing="ij"
    )
    distance = torch.sqrt((xx - x).square() + (yy - y).square())
    circle = torch.where(
        distance < radius,
        ((radius - distance) / radius * 0.7 + 0.3) * gain,
        torch.zeros_like(distance),
    )
    image.copy_(torch.minimum(torch.ones_like(image), image + circle))


def _legacy_flood_reward(image: np.ndarray, allowed: np.ndarray, x: int, y: int) -> np.ndarray:
    height, width = image.shape
    flooded = np.zeros((height, width), dtype=np.uint8)
    quantized = np.rint(image * 100).astype(np.uint8)
    for seed_y in range(max(0, y - 2), min(height - 1, y + 2) + 1):
        for seed_x in range(max(0, x - 2), min(width - 1, x + 2) + 1):
            flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
            cv2.floodFill(quantized.copy(), flood_mask, (seed_x, seed_y), (255,), (0,), (0,))
            flooded |= flood_mask[1:-1, 1:-1]
    return np.bitwise_and(flooded, allowed)


def _trace(index: int, screenshot: str) -> dict[str, object]:
    return {
        "step_id": "testing-0",
        "index": index,
        "action": {"kind": "click", "x": 32, "y": 32},
        "reward": index + 0.5,
        "branch_symbols": [1, 2 + index],
        "network_symbols": [10],
        "viewport": [64, 64],
        "action_targets": [
            {
                "bounds": [10, 10, 40, 40],
                "click": True,
                "right_click": False,
                "type": False,
                "scroll": False,
                "scroll_up": False,
                "scroll_down": False,
            }
        ],
        "screenshot": screenshot,
    }
