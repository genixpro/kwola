"""The two intentionally supported built-in profiles."""

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
        "training": {"batch_size": 4, "batches_per_iteration": 8},
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
            "enable_cursor_prediction": False,
            "enable_execution_feature_prediction": False,
            "enable_trace_prediction": False,
        },
        "training": {
            "batch_size": 24,
            "batches_per_iteration": 4,
            "gradient_beta": 0.9,
            "device_indices": [0, 1],
            "world_size": 2,
        },
        "storage": {"codec_compression_level": 0},
        "reporting": {"chart_every_testing_steps": 5},
    },
}


def profile_config(profile: ProfileName, target: str, seed: int) -> RunConfig:
    """Return a fresh immutable profile; callers can never mutate the cache."""
    data = deepcopy(_PROFILES[profile])
    return RunConfig.model_validate({"target": target, "profile": profile, "seed": seed, **data})
