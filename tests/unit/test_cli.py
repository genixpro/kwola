from pathlib import Path

from kwola.bin.cli import build_parser, main
from kwola.config import load_config
from kwola.storage import LmdbRunStore, load_manifest


def test_cli_exposes_only_the_supported_top_level_commands() -> None:
    parser = build_parser()
    subcommands = next(action for action in parser._actions if getattr(action, "choices", None))

    assert set(subcommands.choices) == {
        "init",
        "run",
        "test-step",
        "train-step",
        "report",
        "doctor",
        "benchmark",
        "proxy",
    }


def test_init_creates_a_complete_fresh_run_layout(tmp_path: Path) -> None:
    run_dir = tmp_path / "fresh-run"

    result = main(
        [
            "init",
            "http://127.0.0.1:3001/",
            "--profile",
            "testing",
            "--run-dir",
            str(run_dir),
            "--seed",
            "41",
        ]
    )

    assert result == 0
    assert load_config(run_dir).seed == 41
    assert load_manifest(run_dir).enabled_browsers[0].value == "chromium"
    assert (run_dir / "blobs").is_dir()
    assert (run_dir / "checkpoints").is_dir()
    with LmdbRunStore(run_dir / "run.lmdb", readonly=True) as store:
        assert store.get("run", "state") == {"testing_steps": 0, "training_steps": 0}


def test_viewport_rejects_malformed_values() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["test-step", "run", "--viewport", "bad"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("malformed viewport was accepted")
