"""Immutable replay plan shared by all distributed training ranks."""

from dataclasses import dataclass
from pathlib import Path

from kwola.config import load_config

from .replay import require_replay_budget
from .replay_state import open_replay_store


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    step_index: int
    training_index: int
    trace_ids: tuple[str, ...]
    iteration_count: int
    replay_sample_credit: int

    @property
    def trace_count(self) -> int:
        return len(self.trace_ids)


def training_plan(run_dir: Path) -> TrainingPlan:
    config = load_config(run_dir)
    with open_replay_store(run_dir, readonly=True) as store:
        state = store.get("run", "state") or {}
        trace_ids = tuple(key for key, _trace in store.scan("traces"))
    requested = int(
        state.get("scheduled_training_iterations", config.training.batches_per_iteration)
    )
    budget = require_replay_budget(
        len(trace_ids) - int(state.get("training_trace_count", 0)),
        requested,
        config.training.batch_size,
        config.training.world_size,
        config.training.replay_samples_per_new_trace,
        int(state.get("replay_sample_credit", 0)),
        len(trace_ids),
        "distributed training",
    )
    return TrainingPlan(
        step_index=int(state.get("training_steps", 0)),
        training_index=int(state.get("training_iterations", 0)),
        trace_ids=trace_ids,
        iteration_count=budget.iterations,
        replay_sample_credit=budget.remaining_sample_credit,
    )
