"""Run summary and reward-chart generation."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as pyplot

from kwola.config import load_config
from kwola.storage import LmdbRunStore

matplotlib.use("Agg")


class ReportService:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)

    def generate(self) -> tuple[Path, ...]:
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
        summary_path = report_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        chart_path = report_dir / "rewards.png"
        self._reward_chart(chart_path, traces)
        return summary_path, chart_path

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


def _reward(trace: dict[str, object]) -> float:
    value = trace["reward"]
    if not isinstance(value, int | float):
        raise TypeError("trace reward is not numeric")
    return float(value)
