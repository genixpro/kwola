"""Persistent replay high-water state shared by training entry points."""

from pathlib import Path

from kwola.config import load_config
from kwola.storage import LmdbRunStore

from .replay import require_replay_budget


def open_replay_store(run_dir: Path, readonly: bool = False) -> LmdbRunStore:
    config = load_config(run_dir)
    return LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        compression_level=config.storage.codec_compression_level,
        readonly=readonly,
    )


def require_new_replay(run_dir: Path) -> None:
    config = load_config(run_dir)
    with open_replay_store(run_dir, readonly=True) as store:
        trace_count = sum(1 for _ in store.scan("traces"))
        state = store.get("run", "state") or {}
    require_replay_budget(
        trace_count - int(state.get("training_trace_count", 0)),
        1,
        config.training.batch_size,
        config.training.world_size,
        config.training.replay_samples_per_new_trace,
        int(state.get("replay_sample_credit", 0)),
        trace_count,
        "training",
    )
