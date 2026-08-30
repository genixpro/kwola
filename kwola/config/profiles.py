"""Fresh built-in profiles for tests, portable runs, and the reference rig."""

from copy import deepcopy
from typing import Any

from .models import ProfileName, RunConfig


def _layers(kernels: tuple[int, ...]) -> list[dict[str, int]]:
    strides = (2, 2, 2, 1, 1)
    return [
        {"kernels": count, "stride": stride} for count, stride in zip(kernels, strides, strict=True)
    ]


_PROFILES: dict[ProfileName, dict[str, Any]] = {
    "testing": {
        "browser": {},
        "instrumentation": {},
        "policy": {"custom_typing_strings": ["action_a", "b_action"]},
        "model": {
            "layers": _layers((16, 24, 32, 48, 64)),
            "pixel_features": 48,
            "additional_stamp_depth": 5,
            "symbol_dictionary_size": 25_000,
            "symbol_embedding_size": 32,
        },
        "training": {
            "batch_size": 4,
            "batches_per_iteration": 10,
            "crop_random_x": 75,
            "crop_random_y": 75,
        },
        "storage": {"codec_compression_level": 1},
        "reporting": {},
    },
    "standard": {
        "browser": {},
        "instrumentation": {},
        "policy": {"exploration": {"max_test_step_index": 300}},
        "model": {
            "layers": _layers((32, 48, 64, 96, 128)),
            "pixel_features": 96,
            "additional_stamp_depth": 10,
            "symbol_dictionary_size": 100_000,
            "symbol_embedding_size": 128,
        },
        "training": {
            "batch_size": 24,
            "batches_per_iteration": 600,
            "batch_iteration_adjustment": 25,
            "gradient_beta": 0.9,
            "device_indices": [0, 1],
            "world_size": 2,
        },
        "storage": {"codec_compression_level": 0},
        "reporting": {"chart_every_testing_steps": 5},
    },
    "rig": {
        "browser": {
            "enabled": ["chromium", "firefox"],
            "viewports": [{"width": 1920, "height": 1080}],
        },
        "instrumentation": {},
        "policy": {
            "exploration": {"max_test_step_index": 300},
            "testing_sequence_length": 50,
        },
        "model": {
            "layers": _layers((32, 48, 64, 96, 128)),
            "pixel_features": 96,
            "additional_stamp_depth": 10,
            "symbol_dictionary_size": 100_000,
            "symbol_embedding_size": 128,
        },
        "training": {
            "batch_size": 48,
            "batches_per_iteration": 600,
            "min_batches_per_iteration": 300,
            "max_batches_per_iteration": 900,
            "batch_iteration_adjustment": 50,
            "gradient_beta": 0.9,
            "device_indices": [0, 1],
            "world_size": 2,
            "cpu_threads_per_rank": 4,
            "batch_prefetch": True,
            "decoded_image_cache_size": 4096,
            "telemetry_every_iterations": 8,
            "checkpoint_every_iterations": 1,
            "sample_cache_workers": 8,
        },
        "storage": {"codec_compression_level": 0},
        "reporting": {
            "chart_every_testing_steps": 25,
            "debug_videos": True,
            "annotated_videos": False,
            "debug_video_every_testing_steps": 25,
        },
        "orchestration": {
            "browser_workers": 8,
            "browser_cpu_threads": 2,
            "telemetry_interval_seconds": 5.0,
            "minimum_traces_before_training": 20,
        },
    },
}


def profile_config(profile: ProfileName, target: str, seed: int) -> RunConfig:
    """Return a fresh immutable profile; callers can never mutate the cache."""
    data = deepcopy(_PROFILES[profile])
    return RunConfig.model_validate({"target": target, "profile": profile, "seed": seed, **data})
