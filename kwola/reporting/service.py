"""Run summary and reward-chart generation."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as pyplot

from kwola.config import load_config
from kwola.storage import AtomicBlobStore, LmdbRunStore

from .videos import VideoRenderer

matplotlib.use("Agg")


class ReportService:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)

    def generate(self, *, scheduled: bool = False) -> tuple[Path, ...]:
        report_dir = self._run_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        traces, testing, training, bugs = self._records()
        summary: dict[str, int | float] = {
            "traces": len(traces),
            "testing_steps": len(testing),
            "training_steps": len(training),
            "bugs": len(bugs),
            "total_reward": sum(_reward(trace) for trace in traces),
        }
        blob_store = AtomicBlobStore(self._run_dir)
        summary_path = blob_store.write(
            "reports", "summary.json", (json.dumps(summary, indent=2) + "\n").encode()
        )
        artifacts = [summary_path]
        chart_due = len(testing) % self._config.reporting.chart_every_testing_steps == 0
        if self._config.reporting.charts and (chart_due or not scheduled):
            chart_path = report_dir / "rewards.png"
            self._reward_chart(chart_path, traces)
            artifacts.append(chart_path)
        artifacts.extend(self._videos(traces))
        artifacts.extend(self._bug_reports(bugs, blob_store))
        return tuple(artifacts)

    def _records(
        self,
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
    ]:
        with LmdbRunStore(
            self._run_dir / self._config.storage.database_directory,
            map_size=self._config.storage.database_map_size_bytes,
            readonly=True,
        ) as store:
            return tuple(  # type: ignore[return-value]
                [record for _key, record in store.scan(collection)]
                for collection in ("traces", "testing_steps", "training_steps", "bugs")
            )

    @staticmethod
    def _reward_chart(path: Path, traces: list[dict[str, object]]) -> None:
        figure, axes = pyplot.subplots(figsize=(10, 4))
        axes.plot([_reward(trace) for trace in traces])
        axes.set_title("Reward by trace")
        axes.set_xlabel("Trace")
        axes.set_ylabel("Present reward")
        figure.tight_layout()
        figure.savefig(path)
        pyplot.close(figure)

    def _videos(self, traces: list[dict[str, object]]) -> list[Path]:
        if not (self._config.reporting.debug_videos or self._config.reporting.annotated_videos):
            return []
        groups: dict[str, list[dict[str, object]]] = {}
        for trace in traces:
            groups.setdefault(str(trace["step_id"]), []).append(trace)
        renderer = VideoRenderer(
            self._run_dir, timeout_seconds=self._config.reporting.video_timeout_seconds
        )
        artifacts = []
        for step_id, step_traces in groups.items():
            ordered = sorted(step_traces, key=_trace_index)
            if self._config.reporting.debug_videos:
                path = self._run_dir / "reports" / "videos" / f"{step_id}-debug.mp4"
                artifacts.append(renderer.render(path, ordered, annotated=False))
            if self._config.reporting.annotated_videos:
                path = self._run_dir / "reports" / "videos" / f"{step_id}-annotated.mp4"
                artifacts.append(renderer.render(path, ordered, annotated=True))
        return artifacts

    def _bug_reports(self, bugs: list[dict[str, object]], blobs: AtomicBlobStore) -> list[Path]:
        if not self._config.reporting.bug_reports:
            return []
        artifacts = []
        for bug in bugs:
            fingerprint = str(bug["fingerprint"])
            payload = json.dumps(bug, indent=2, sort_keys=True).encode() + b"\n"
            artifacts.append(blobs.write("reports/bugs", f"{fingerprint}.json", payload))
        return artifacts


def _reward(trace: dict[str, object]) -> float:
    value = trace["reward"]
    if not isinstance(value, int | float):
        raise TypeError("trace reward is not numeric")
    return float(value)


def _trace_index(trace: dict[str, object]) -> int:
    value = trace["index"]
    if not isinstance(value, int):
        raise TypeError("trace index is not an integer")
    return value
