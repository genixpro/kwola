"""Credit-budgeted shuffled replay with disjoint distributed-rank slices."""

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReplayBudget:
    """Optimizer work earned by fresh traces and retained across training steps."""

    iterations: int
    remaining_sample_credit: int


def minimum_replay_size(batch_size: int, world_size: int) -> int:
    """Return the samples required for one duplicate-free global update."""
    return batch_size * world_size


def replay_budget(
    new_trace_count: int,
    requested: int,
    batch_size: int,
    world_size: int,
    samples_per_new_trace: int,
    carried_sample_credit: int = 0,
) -> ReplayBudget:
    """Convert new traces into replay work without discarding partial batches."""
    if new_trace_count < 0:
        raise ValueError("new trace count cannot be negative")
    if carried_sample_credit < 0:
        raise ValueError("replay sample credit cannot be negative")
    if samples_per_new_trace < 1:
        raise ValueError("replay samples per new trace must be positive")
    global_batch_size = minimum_replay_size(batch_size, world_size)
    available = carried_sample_credit + new_trace_count * samples_per_new_trace
    complete_global_batches = available // global_batch_size
    iterations = min(requested, complete_global_batches)
    return ReplayBudget(iterations, available - iterations * global_batch_size)


def require_replay_budget(
    new_trace_count: int,
    requested: int,
    batch_size: int,
    world_size: int,
    samples_per_new_trace: int,
    carried_sample_credit: int,
    replay_size: int,
    label: str = "training",
) -> ReplayBudget:
    """Return earned work and its residual credit, requiring a viable replay buffer."""
    required = minimum_replay_size(batch_size, world_size)
    if replay_size < required:
        raise RuntimeError(f"{label} requires at least {required} replay traces")
    budget = replay_budget(
        new_trace_count,
        requested,
        batch_size,
        world_size,
        samples_per_new_trace,
        carried_sample_credit,
    )
    if budget.iterations:
        return budget
    raise RuntimeError(f"{label} requires {required} replay-sample credits for one global batch")


class ReplaySampler:
    def __init__(
        self,
        size: int,
        batch_size: int,
        world_size: int,
        rank: int,
        seed: int,
        training_step: int,
    ) -> None:
        if size < 1:
            raise ValueError("replay requires at least one trace")
        if not 0 <= rank < world_size:
            raise ValueError("replay rank must be within world size")
        self._size = size
        self._batch_size = batch_size
        self._world_size = world_size
        self._rank = rank
        self._global_batch_size = batch_size * world_size
        self._seed = seed
        self._training_step = training_step
        self._permutations: dict[int, tuple[int, ...]] = {}

    def batch_indexes(self, iteration: int) -> tuple[int, ...]:
        global_start = (iteration * self._world_size + self._rank) * self._batch_size
        return tuple(self._index(global_start + offset) for offset in range(self._batch_size))

    def _index(self, position: int) -> int:
        epoch, offset = divmod(position, self._size)
        self._ensure_permutations(epoch)
        return self._permutations[epoch][offset]

    def _ensure_permutations(self, requested_epoch: int) -> None:
        for epoch in range(len(self._permutations), requested_epoch + 1):
            values = list(range(self._size))
            mixed_seed = (
                self._seed * 1_000_003 + self._training_step * 97_409 + epoch * 65_537
            ) & (2**63 - 1)
            random.Random(mixed_seed).shuffle(values)
            boundary_offset = (epoch * self._size) % self._global_batch_size
            if epoch and self._size >= self._global_batch_size and boundary_offset:
                previous = self._permutations[epoch - 1]
                forbidden = frozenset(previous[-boundary_offset:])
                values = [value for value in values if value not in forbidden] + [
                    value for value in values if value in forbidden
                ]
            self._permutations[epoch] = tuple(values)
