"""Configuration-backed recorded-sample assembler construction."""

import random
from collections.abc import Sequence
from pathlib import Path

from kwola.agent import action_catalog
from kwola.config.models import RunConfig
from kwola.storage import LmdbRunStore

from .samples import RecordedSampleAssembler


def recorded_sample_assembler(
    run_dir: Path,
    store: LmdbRunStore,
    config: RunConfig,
    *,
    compact_cpu_tensors: bool = False,
    freeze_records: bool = False,
    trace_ids: Sequence[str] | None = None,
    random_seed: int | None = None,
) -> RecordedSampleAssembler:
    return RecordedSampleAssembler(
        run_dir,
        store,
        symbol_dictionary_size=config.model.symbol_dictionary_size,
        discount_rate=config.policy.rewards.discount_rate,
        max_discounted_reward=config.policy.rewards.max_discounted_reward,
        cache_version=config.training.sample_cache_version,
        channels=action_catalog(config.policy),
        recent_action_history=config.model.recent_action_history,
        recent_action_radius=config.training.recent_action_image_radius,
        recent_action_decay=config.training.recent_action_image_decay,
        image_downscale_ratio=config.model.image_downscale_ratio,
        crop_size=(config.training.crop_width, config.training.crop_height),
        next_crop_size=(config.training.next_crop_width, config.training.next_crop_height),
        crop_random=(config.training.crop_random_x, config.training.crop_random_y),
        decoded_image_cache_size=config.training.decoded_image_cache_size,
        decoded_image_cache_directory=run_dir / config.storage.cache_directory / "decoded-images",
        compact_cpu_tensors=compact_cpu_tensors,
        freeze_records=freeze_records,
        trace_ids=trace_ids,
        enable_trace_prediction=config.model.enable_trace_prediction,
        enable_execution_feature_prediction=config.model.enable_execution_feature_prediction,
        enable_cursor_prediction=config.model.enable_cursor_prediction,
        source=random.Random(config.seed if random_seed is None else random_seed),
    )
