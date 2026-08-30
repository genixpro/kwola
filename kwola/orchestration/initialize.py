"""Fresh-run initialization for the clean-break schema."""

from pathlib import Path

from kwola.config import create_run_config
from kwola.config.models import ProfileName
from kwola.storage import LmdbRunStore, RunManifest, save_manifest


def initialize_run(target: str, profile: ProfileName, run_dir: Path, seed: int) -> RunManifest:
    config = create_run_config(target, profile, run_dir, seed)
    manifest = RunManifest.create(
        target=config.target,
        profile=config.profile,
        seed=config.seed,
        enabled_browsers=config.browser.enabled,
    )
    save_manifest(manifest, run_dir)
    for directory in (
        config.storage.blobs_directory,
        config.storage.cache_directory,
        config.storage.checkpoints_directory,
        "logs",
        "reports",
    ):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    with LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        compression_level=config.storage.codec_compression_level,
    ) as store:
        store.put("run", "state", {"testing_steps": 0, "training_steps": 0})
    return manifest
