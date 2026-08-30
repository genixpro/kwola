"""Supervised alternating testing/training experiment loop."""

from pathlib import Path

from .testing import TestingRunner
from .training import TrainingRunner


class ExperimentRunner:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def run(self) -> int:
        try:
            while True:
                TestingRunner(self._run_dir).run()
                TrainingRunner(self._run_dir).run()
        except KeyboardInterrupt:
            return 130
