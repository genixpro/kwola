"""Persistent replay high-water state shared by training entry points."""

from pathlib import Path

from kwola.config import load_config
from kwola.storage import LmdbRunStore

from .replay import require_replay_iterations


def require_new_replay(run_dir: Path) -> None:
    config = load_config(run_dir)
    with LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        compression_level=config.storage.codec_compression_level,
        readonly=True,
    ) as store:
        trace_count = sum(1 for _ in store.scan("traces"))
        state = store.get("run", "state") or {}
    require_replay_iterations(
        trace_count - int(state.get("training_trace_count", 0)),
        1,
        config.training.batch_size,
        config.training.world_size,
        "training",
    )
