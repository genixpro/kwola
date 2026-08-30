"""Deterministic shuffled replay with disjoint distributed-rank slices."""

import random


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
