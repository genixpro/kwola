"""Ordered built-in hooks for core testing, training, and reporting concerns."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kwola.config.models import RunConfig
from kwola.storage import LmdbRunStore

from .events import LifecycleEvent, LifecycleEventName
from .registry import LifecycleHook


@dataclass(slots=True)
class CoreHook:
    name: str
    order: int
    fatal: bool
    events: frozenset[LifecycleEventName]
    callback: Callable[[LifecycleEvent], None]

    def handle(self, event: LifecycleEvent) -> None:
        self.callback(event)

    def close(self) -> None:
        return None


def testing_core_hooks(run_dir: Path, config: RunConfig) -> tuple[LifecycleHook, ...]:
    hooks: list[LifecycleHook] = [
        CoreHook(
            "telemetry",
            10,
            False,
            frozenset({LifecycleEventName.AFTER_ACTION}),
            lambda event: _record_metric(run_dir, config, event, "telemetry"),
        ),
        CoreHook(
            "screenshots",
            20,
            True,
            frozenset({LifecycleEventName.TRACE_RECORDED}),
            lambda event: _audit_screenshots(run_dir, event),
        ),
        CoreHook(
            "bugs",
            30,
            False,
            frozenset({LifecycleEventName.TRACE_RECORDED}),
            _audit_bugs,
        ),
        CoreHook(
            "sample-precomputation",
            40,
            True,
            frozenset({LifecycleEventName.SESSION_FINISHED}),
            _invoke_sample_precomputation,
        ),
        CoreHook(
            "metrics",
            50,
            False,
            frozenset({LifecycleEventName.TRACE_RECORDED}),
            lambda event: _record_metric(run_dir, config, event, "trace"),
        ),
    ]
    if config.reporting.debug_videos:
        hooks.append(
            CoreHook(
                "videos",
                60,
                False,
                frozenset({LifecycleEventName.SESSION_FINISHED}),
                lambda event: _generate_debug_video(run_dir, config, event),
            )
        )
    if config.reporting.annotated_videos:
        hooks.append(
            CoreHook(
                "annotated-videos",
                70,
                False,
                frozenset({LifecycleEventName.RUN_FINISHED}),
                lambda _event: _generate_reports(run_dir),
            )
        )
    return tuple(hooks)


def training_core_hooks(run_dir: Path, config: RunConfig) -> tuple[LifecycleHook, ...]:
    return (
        CoreHook(
            "metrics",
            10,
            False,
            frozenset({LifecycleEventName.TRAINING_ITERATION_FINISHED}),
            lambda event: _record_metric(run_dir, config, event, "training"),
        ),
    )


def _audit_screenshots(run_dir: Path, event: LifecycleEvent) -> None:
    trace = _trace(event)
    for key in ("screenshot_before", "screenshot"):
        path = run_dir / str(trace[key])
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"recorded trace has invalid {key}: {path}")


def _audit_bugs(event: LifecycleEvent) -> None:
    trace = _trace(event)
    store = _event_store(event)
    for message in trace.get("errors", []):
        import hashlib

        fingerprint = hashlib.sha256(str(message).encode()).hexdigest()
        if store.get("bugs", fingerprint) is None:
            raise ValueError(f"bug record was not persisted for {fingerprint}")


def _invoke_sample_precomputation(event: LifecycleEvent) -> None:
    callback = event.values().get("prepare_samples")
    if not callable(callback):
        raise ValueError("sample precomputation callback is missing")
    callback()


def _trace(event: LifecycleEvent) -> dict[str, Any]:
    if event.subject_id is None:
        raise ValueError("trace hook event is missing its subject id")
    trace = _event_store(event).get("traces", event.subject_id)
    if trace is None:
        raise ValueError(f"trace record is missing: {event.subject_id}")
    return trace


def _event_store(event: LifecycleEvent) -> LmdbRunStore:
    store = event.values().get("store")
    if not isinstance(store, LmdbRunStore):
        raise ValueError("core hook event is missing its run store")
    return store


def _record_metric(
    run_dir: Path,
    config: RunConfig,
    event: LifecycleEvent,
    category: str,
) -> None:
    values = {key: value for key, value in event.payload if key != "store"}
    key = f"{category}:{event.subject_id or event.occurred_at}"
    supplied = event.values().get("store")
    if isinstance(supplied, LmdbRunStore):
        supplied.put("hook_metrics", key, values)
        return
    with LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        compression_level=config.storage.codec_compression_level,
    ) as store:
        store.put("hook_metrics", key, values)


def _generate_reports(run_dir: Path) -> None:
    from kwola.reporting.service import ReportService

    ReportService(run_dir).generate(scheduled=True)


def _generate_debug_video(
    run_dir: Path,
    config: RunConfig,
    event: LifecycleEvent,
) -> None:
    from kwola.agent import InferenceDiagnostics
    from kwola.reporting import RichDebugVideoRenderer

    if event.subject_id is None:
        raise ValueError("debug video event is missing its testing step id")
    supplied = event.values().get("diagnostics")
    if not isinstance(supplied, tuple) or not supplied:
        return
    if not all(value is None or isinstance(value, InferenceDiagnostics) for value in supplied):
        raise ValueError("debug video event has invalid inference diagnostics")
    store = _event_store(event)
    traces = [record for _key, record in store.scan_prefix("traces", event.subject_id)]
    traces.sort(key=lambda trace: int(trace["index"]))
    if len(traces) != len(supplied):
        raise ValueError("debug video traces and inference diagnostics are not aligned")
    relative = Path("reports") / "videos" / f"{event.subject_id}-debug.mp4"
    RichDebugVideoRenderer(run_dir, config).render(run_dir / relative, traces, supplied)

    def attach_video(current: dict[str, Any] | None) -> dict[str, Any]:
        record = dict(current or {})
        record["debug_video"] = str(relative)
        return record

    store.update("testing_steps", event.subject_id, attach_video)
