"""CPU batch assembly streams that can overlap work with GPU optimization."""

import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from .ddp import DistributedCoordinator
from .replay import ReplaySampler
from .samples import RecordedSampleAssembler, TrainingBatch
from .spool import pin_batch


def batches(
    assembler: RecordedSampleAssembler,
    coordinator: DistributedCoordinator,
    count: int,
    initial: TrainingBatch | None,
    batch_size: int,
    sampler: ReplaySampler,
    impossible_reward: float,
    prefetch: bool,
    pin_memory: bool = False,
) -> Iterator[tuple[TrainingBatch, float]]:
    if not prefetch:
        for iteration in range(count):
            if iteration == 0 and initial is not None:
                yield initial, 0.0
            else:
                yield _assemble(
                    assembler,
                    coordinator,
                    iteration,
                    batch_size,
                    sampler,
                    impossible_reward,
                    pin_memory,
                )
        return
    yield from _prefetched(
        assembler,
        coordinator,
        count,
        initial,
        batch_size,
        sampler,
        impossible_reward,
        pin_memory,
    )


def _prefetched(
    assembler: RecordedSampleAssembler,
    coordinator: DistributedCoordinator,
    count: int,
    initial: TrainingBatch | None,
    batch_size: int,
    sampler: ReplaySampler,
    impossible_reward: float,
    pin_memory: bool,
) -> Iterator[tuple[TrainingBatch, float]]:
    future: Future[tuple[TrainingBatch, float]] | None = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="kwola-batch") as executor:
        if initial is None:
            future = executor.submit(
                _assemble,
                assembler,
                coordinator,
                0,
                batch_size,
                sampler,
                impossible_reward,
                pin_memory,
            )
        for iteration in range(count):
            if iteration == 0 and initial is not None:
                batch = (initial, 0.0)
            else:
                assert future is not None
                batch = future.result()
            if iteration + 1 < count:
                future = executor.submit(
                    _assemble,
                    assembler,
                    coordinator,
                    iteration + 1,
                    batch_size,
                    sampler,
                    impossible_reward,
                    pin_memory,
                )
            yield batch


def _assemble(
    assembler: RecordedSampleAssembler,
    coordinator: DistributedCoordinator,
    iteration: int,
    batch_size: int,
    sampler: ReplaySampler,
    impossible_reward: float,
    pin_memory: bool,
) -> tuple[TrainingBatch, float]:
    started = time.perf_counter()
    batch = assembler.assemble(
        batch_size=batch_size,
        device=torch.device("cpu"),
        impossible_reward=impossible_reward,
        sample_indexes=sampler.batch_indexes(iteration),
    )
    if pin_memory:
        batch = pin_batch(batch)
    return batch, time.perf_counter() - started
