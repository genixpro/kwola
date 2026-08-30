"""Atomic configuration loading and run-directory creation."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .models import ProfileName, RunConfig
from .profiles import profile_config

CONFIG_NAME = "kwola.json"


def load_config(run_dir: Path) -> RunConfig:
    path = run_dir / CONFIG_NAME
    with path.open("rb") as stream:
        return RunConfig.model_validate_json(stream.read())


def save_config(config: RunConfig, run_dir: Path) -> Path:
    _reject_inline_secrets(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / CONFIG_NAME
    _atomic_json_write(target, config.model_dump(mode="json"))
    return target


def _reject_inline_secrets(config: RunConfig) -> None:
    login = config.browser.autologin
    actions = config.policy.actions
    if any((login.email, login.password, actions.email, actions.password)):
        raise ValueError(
            "inline credentials cannot be persisted; configure *_environment fields instead"
        )


def create_run_config(
    target: str,
    profile: ProfileName,
    run_dir: Path,
    seed: int,
    overrides: dict[str, Any] | None = None,
) -> RunConfig:
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    config = profile_config(profile, target, seed)
    if overrides:
        config = RunConfig.model_validate(_deep_merge(config.model_dump(), overrides))
    save_config(config, run_dir)
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _atomic_json_write(path: Path, data: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
