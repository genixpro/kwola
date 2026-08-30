from pathlib import Path

import pytest
from pydantic import ValidationError

from kwola.config import create_run_config, load_config, profile_config


def test_profiles_are_fresh_and_preserve_expected_topologies() -> None:
    first = profile_config("testing", "http://127.0.0.1:3001/", 7)
    second = profile_config("testing", "http://127.0.0.1:3001/", 7)
    standard = profile_config("standard", "http://127.0.0.1:3003/", 8)
    rig = profile_config("rig", "http://127.0.0.1:3003/", 9)

    assert first is not second
    assert [layer.kernels for layer in first.model.layers] == [16, 24, 32, 48, 64]
    assert [layer.kernels for layer in standard.model.layers] == [32, 48, 64, 96, 128]
    assert first.training.batch_size == 4
    assert standard.training.batch_size == 24
    assert rig.orchestration.browser_workers == 8
    assert rig.policy.testing_sequence_length == 50
    assert rig.training.batch_prefetch
    assert rig.training.batches_per_iteration == 600
    assert rig.training.cpu_threads_per_rank == 4
    assert rig.training.decoded_image_cache_size == 4096
    assert rig.orchestration.browser_cpu_threads == 2


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


@pytest.mark.parametrize(
    "origin",
    (
        "https://user@example.net",
        "https://example.net/path",
        "https://example.net?query=yes",
        "https://example.net#fragment",
        "data:text/html,invalid",
    ),
)
def test_allowed_navigation_origins_reject_non_origins(origin: str) -> None:
    config = profile_config("testing", "https://example.com", 1).model_dump()
    config["browser"]["allowed_navigation_origins"] = [origin]

    with pytest.raises(ValidationError):
        type(profile_config("testing", "https://example.com", 1)).model_validate(config)


def test_config_round_trip_is_atomic_and_directory_must_be_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    created = create_run_config("https://example.com", "testing", run_dir, 123)

    assert load_config(run_dir) == created
    assert not tuple(run_dir.glob(".kwola.json.*"))
    with pytest.raises(FileExistsError):
        create_run_config("https://example.com", "testing", run_dir, 123)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda data: data["browser"].update(enabled=[]), "at least one browser"),
        (
            lambda data: data["browser"].update(enabled=["chromium", "chromium"]),
            "unique",
        ),
        (lambda data: data["browser"].update(viewports=[]), "at least one viewport"),
        (lambda data: data["model"].update(layers=data["model"]["layers"][:4]), "five"),
        (
            lambda data: data["model"]["layers"][3].update(kernels=999),
            "pixel_features",
        ),
        (
            lambda data: data["training"].update(world_size=2, device_indices=[]),
            "CPU training",
        ),
        (
            lambda data: data["instrumentation"].update(enabled=False, rewrite_javascript=True),
            "requires instrumentation",
        ),
    ),
)
def test_all_cross_field_constraints_are_actionable(mutation: object, message: str) -> None:
    config = profile_config("testing", "https://example.com", 1)
    data = config.model_dump()
    assert callable(mutation)
    mutation(data)
    with pytest.raises(ValidationError, match=message):
        type(config).model_validate(data)
