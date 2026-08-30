from pathlib import Path

import pytest
from pydantic import ValidationError

from kwola.config import create_run_config, load_config, profile_config


def test_profiles_are_fresh_and_preserve_expected_topologies() -> None:
    first = profile_config("testing", "http://127.0.0.1:3001/", 7)
    second = profile_config("testing", "http://127.0.0.1:3001/", 7)
    standard = profile_config("standard", "http://127.0.0.1:3003/", 8)

    assert first is not second
    assert [layer.kernels for layer in first.model.layers] == [16, 24, 32, 48, 64]
    assert [layer.kernels for layer in standard.model.layers] == [32, 48, 64, 96, 128]
    assert first.training.batch_size == 4
    assert standard.training.batch_size == 24


def test_unknown_keys_are_rejected() -> None:
    config = profile_config("testing", "https://example.com", 1).model_dump()
    config["legacy_flat_key"] = True

    with pytest.raises(ValidationError, match="legacy_flat_key"):
        type(profile_config("testing", "https://example.com", 1)).model_validate(config)


def test_cross_field_validation_rejects_invalid_ddp() -> None:
    config = profile_config("testing", "https://example.com", 1).model_dump()
    config["training"]["device_indices"] = [0, 1]
    config["training"]["world_size"] = 1

    with pytest.raises(ValidationError, match="world_size"):
        type(profile_config("testing", "https://example.com", 1)).model_validate(config)


def test_cross_field_validation_rejects_enabled_login_without_credentials() -> None:
    config = profile_config("testing", "https://example.com", 1).model_dump()
    config["browser"]["autologin"] = {"enabled": True}

    with pytest.raises(ValidationError, match="autologin"):
        type(profile_config("testing", "https://example.com", 1)).model_validate(config)


def test_config_round_trip_is_atomic_and_directory_must_be_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    created = create_run_config("https://example.com", "testing", run_dir, 123)

    assert load_config(run_dir) == created
    assert not tuple(run_dir.glob(".kwola.json.*"))
    with pytest.raises(FileExistsError):
        create_run_config("https://example.com", "testing", run_dir, 123)
