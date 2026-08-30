import hashlib
from pathlib import Path

import pytest

from kwola.config import load_config
from kwola.hooks import (
    HookExecutionError,
    HookRegistry,
    LifecycleEvent,
    LifecycleEventName,
    training_core_hooks,
)
from kwola.hooks import testing_core_hooks as build_testing_core_hooks
from kwola.orchestration.initialize import initialize_run
from kwola.storage import LmdbRunStore


def test_testing_core_hooks_audit_and_precompute(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 1)
    config = load_config(tmp_path)
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    selected = tuple(
        hook for hook in build_testing_core_hooks(tmp_path, config) if hook.name != "videos"
    )
    registry = HookRegistry(selected)
    called: list[bool] = []
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        store.put(
            "traces",
            "trace",
            {"screenshot_before": "before.png", "screenshot": "after.png", "errors": []},
        )
        registry.dispatch(_event(LifecycleEventName.AFTER_ACTION, store, "trace"))
        registry.dispatch(_event(LifecycleEventName.TRACE_RECORDED, store, "trace"))
        registry.dispatch(
            LifecycleEvent(
                LifecycleEventName.SESSION_FINISHED,
                1.0,
                "run",
                "session",
                (("store", store), ("prepare_samples", lambda: called.append(True))),
            )
        )
        assert called == [True]
        assert len(list(store.scan("hook_metrics"))) == 2
    assert registry.close() == ()


def test_core_screenshot_hook_is_fatal_and_training_metrics_open_store(
    tmp_path: Path,
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 2)
    config = load_config(tmp_path)
    selected = tuple(
        hook for hook in build_testing_core_hooks(tmp_path, config) if hook.name == "screenshots"
    )
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        store.put(
            "traces",
            "trace",
            {"screenshot_before": "missing", "screenshot": "missing", "errors": []},
        )
        with pytest.raises(HookExecutionError, match="screenshots"):
            HookRegistry(selected).dispatch(
                _event(LifecycleEventName.TRACE_RECORDED, store, "trace")
            )
    training = HookRegistry(training_core_hooks(tmp_path, config))
    training.dispatch(
        LifecycleEvent(
            LifecycleEventName.TRAINING_ITERATION_FINISHED,
            2.0,
            "run",
            "training-1",
            (("loss", 1.5),),
        )
    )
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2, readonly=True) as store:
        assert store.get("hook_metrics", "training:training-1") == {"loss": 1.5}


def test_core_bug_and_precompute_failure_contracts(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 3)
    config = load_config(tmp_path)
    hooks = build_testing_core_hooks(tmp_path, config)
    bugs = HookRegistry(tuple(hook for hook in hooks if hook.name == "bugs"))
    precompute = HookRegistry(tuple(hook for hook in hooks if hook.name == "sample-precomputation"))
    message = "failure"
    fingerprint = hashlib.sha256(message.encode()).hexdigest()
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        store.put(
            "traces",
            "trace",
            {"screenshot_before": "before", "screenshot": "after", "errors": [message]},
        )
        store.put("bugs", fingerprint, {"fingerprint": fingerprint})
        assert bugs.dispatch(_event(LifecycleEventName.TRACE_RECORDED, store, "trace")) == ()
        store.delete("bugs", fingerprint)
        failures = bugs.dispatch(_event(LifecycleEventName.TRACE_RECORDED, store, "trace"))
        assert len(failures) == 1 and failures[0].hook == "bugs"
        with pytest.raises(HookExecutionError, match="sample-precomputation"):
            precompute.dispatch(_event(LifecycleEventName.SESSION_FINISHED, store, "session"))


def test_core_hook_reports_missing_event_context(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 4)
    config = load_config(tmp_path)
    screenshots = HookRegistry(
        tuple(
            hook
            for hook in build_testing_core_hooks(tmp_path, config)
            if hook.name == "screenshots"
        )
    )
    missing_subject = LifecycleEvent(LifecycleEventName.TRACE_RECORDED, 1.0, "run", None, ())
    with pytest.raises(HookExecutionError, match="subject id"):
        screenshots.dispatch(missing_subject)


def _event(name: LifecycleEventName, store: LmdbRunStore, subject: str) -> LifecycleEvent:
    return LifecycleEvent(name, 1.0, "run", subject, (("store", store), ("value", 1)))
