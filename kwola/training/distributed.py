"""Two-rank recorded-sample training with explicit DDP ownership."""

import copy
import multiprocessing
import random
import socket
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import torch
from torch import distributed
from torch.multiprocessing import spawn  # type: ignore[attr-defined]
from torch.nn.parallel import DistributedDataParallel

from kwola.agent import TraceNet, action_catalog
from kwola.config import load_config
from kwola.orchestration.results import RunnerResult
from kwola.storage import CheckpointPublisher, LmdbRunStore, load_manifest

from .ddp import DistributedCoordinator, DistributedSettings
from .optimizer import ModelOptimizer, OptimizerMetrics
from .samples import RecordedSampleAssembler, TrainingBatch
from .spool import batch_to_device, share_batch


def run_distributed_training(run_dir: Path) -> RunnerResult:
    config = load_config(run_dir)
    devices = config.training.device_indices
    if len(devices) != config.training.world_size:
        raise RuntimeError("distributed training requires one device index per rank")
    if torch.cuda.device_count() < config.training.world_size:
        raise RuntimeError("insufficient CUDA devices for configured world size")
    _prepare_cache(run_dir)
    shared_batches = (
        _shared_initial_batches(run_dir) if config.training.use_shared_memory_spool else ()
    )
    context = multiprocessing.get_context("spawn")
    results: Any = context.SimpleQueue()
    init_method = f"tcp://127.0.0.1:{_free_port()}"
    spawn(  # type: ignore[no-untyped-call]
        _training_rank,
        args=(
            config.training.world_size,
            devices,
            init_method,
            run_dir,
            results,
            shared_batches,
        ),
        nprocs=config.training.world_size,
        join=True,
    )
    return RunnerResult.model_validate_json(results.get())


def _prepare_cache(run_dir: Path) -> None:
    with _store(run_dir) as store:
        assembler = _assembler(run_dir, store)
        config = load_config(run_dir)
        if assembler.prepare_cache(config.training.sample_cache_workers) == 0:
            raise RuntimeError("training requires at least one recorded browser trace")


def _shared_initial_batches(run_dir: Path) -> tuple[TrainingBatch, ...]:
    config = load_config(run_dir)
    batches = []
    with _store(run_dir, readonly=True) as store:
        assembler = _assembler(run_dir, store)
        for rank in range(config.training.world_size):
            batch = assembler.assemble(
                batch_size=config.training.batch_size,
                device=torch.device("cpu"),
                impossible_reward=config.policy.rewards.impossible_action,
                offset=rank * config.training.batch_size,
            )
            batches.append(share_batch(batch))
    return tuple(batches)


def _training_rank(
    rank: int,
    world_size: int,
    devices: tuple[int, ...],
    init_method: str,
    run_dir: Path,
    results: Any,
    shared_batches: tuple[TrainingBatch, ...],
) -> None:
    started = time.perf_counter()
    settings = DistributedSettings(rank, world_size, devices[rank], init_method)
    with DistributedCoordinator(settings) as coordinator:
        initial = shared_batches[rank] if shared_batches else None
        metrics, model, optimizer, step_index, iterations = _rank_step(
            run_dir, coordinator, initial
        )
        loss = torch.tensor([metrics.loss], device=coordinator.device)
        distributed.all_reduce(loss, op=distributed.ReduceOp.SUM)
        coordinator.barrier()
        if coordinator.is_publisher:
            result = _publish_result(
                run_dir,
                model,
                optimizer,
                step_index,
                metrics,
                float(loss.item() / world_size),
                time.perf_counter() - started,
                iterations,
            )
            results.put(result.model_dump_json())


def _rank_step(
    run_dir: Path,
    coordinator: DistributedCoordinator,
    initial_batch: TrainingBatch | None = None,
) -> tuple[OptimizerMetrics, TraceNet, ModelOptimizer, int, int]:
    config = load_config(run_dir)
    manifest = load_manifest(run_dir)
    with _store(run_dir, readonly=True) as store:
        state = store.get("run", "state") or {}
        step_index = int(state.get("training_steps", 0))
        training_index = int(state.get("training_iterations", 0))
        iteration_count = int(
            state.get("scheduled_training_iterations", config.training.batches_per_iteration)
        )
    model = TraceNet(config.model, num_actions=len(action_catalog(config.policy))).to(
        coordinator.device
    )
    checkpoint = manifest.checkpoint.file if manifest.checkpoint else None
    payload = _load_model(run_dir, checkpoint, model, coordinator.device)
    target = copy.deepcopy(model).to(coordinator.device).eval()
    parallel = DistributedDataParallel(
        model,
        device_ids=[coordinator.settings.local_device],
        output_device=coordinator.settings.local_device,
    )
    optimizer = ModelOptimizer(parallel, config.training)
    if payload is not None:
        optimizer.optimizer.load_state_dict(payload["optimizer"])
    metrics = _rank_iterations(
        run_dir,
        coordinator,
        model,
        target,
        optimizer,
        training_index,
        iteration_count,
        initial_batch,
    )
    return metrics, model, optimizer, step_index, iteration_count


def _rank_iterations(
    run_dir: Path,
    coordinator: DistributedCoordinator,
    model: TraceNet,
    target: TraceNet,
    optimizer: ModelOptimizer,
    training_index: int,
    iteration_count: int,
    initial_batch: TrainingBatch | None,
) -> OptimizerMetrics:
    config = load_config(run_dir)
    results = []
    count = iteration_count
    for iteration in range(count):
        batch = (
            batch_to_device(initial_batch, coordinator.device)
            if iteration == 0 and initial_batch is not None
            else _rank_batch(run_dir, coordinator, iteration)
        )
        index = training_index + iteration
        results.append(
            optimizer.step_training(
                batch,
                target,
                index,
                config.policy.rewards.discount_rate,
                config.policy.rewards.max_discounted_reward,
            )
        )
        if (index + 1) % config.training.target_network_update_every == 0:
            target.load_state_dict(model.state_dict())
    duration = sum(result.duration_seconds for result in results)
    samples = count * config.training.batch_size
    return OptimizerMetrics(
        sum(result.loss for result in results) / count,
        duration,
        samples / duration,
    )


def _rank_batch(
    run_dir: Path, coordinator: DistributedCoordinator, iteration: int
) -> TrainingBatch:
    config = load_config(run_dir)
    with _store(run_dir, readonly=True) as store:
        return _assembler(run_dir, store).assemble(
            batch_size=config.training.batch_size,
            device=coordinator.device,
            impossible_reward=config.policy.rewards.impossible_action,
            offset=(coordinator.settings.rank + iteration * config.training.world_size)
            * config.training.batch_size,
        )


def _publish_result(
    run_dir: Path,
    model: TraceNet,
    optimizer: ModelOptimizer,
    step_index: int,
    metrics: OptimizerMetrics,
    mean_loss: float,
    duration: float,
    iterations: int,
) -> RunnerResult:
    config = load_config(run_dir)
    manifest = load_manifest(run_dir)
    publisher = CheckpointPublisher(run_dir, config.storage.checkpoints_directory)
    published = None
    if (step_index + 1) % config.training.checkpoint_every_iterations == 0:
        published = publisher.publish(
            rank=0,
            generation=step_index + 1,
            writer=lambda stream: torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.optimizer.state_dict()},
                stream,
            ),
            manifest=manifest,
            now=time.time(),
        )
    _record_step(
        run_dir,
        step_index,
        mean_loss,
        metrics.duration_seconds,
        iterations,
    )
    return RunnerResult(
        status="completed",
        step_id=f"training-{step_index:08d}",
        duration_seconds=duration,
        artifacts=(str(published[0].relative_to(run_dir)),) if published else (),
        metrics={
            "loss": mean_loss,
            "optimizer_seconds": metrics.duration_seconds,
            "samples_per_second": metrics.samples_per_second * config.training.world_size,
        },
    )


def _load_model(
    run_dir: Path,
    checkpoint: str | None,
    model: TraceNet,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    payload: dict[str, Any] = torch.load(run_dir / checkpoint, map_location=device)
    model.load_state_dict(payload["model"])
    return payload


def _record_step(run_dir: Path, index: int, loss: float, duration: float, iterations: int) -> None:
    with _store(run_dir) as store:
        state = store.get("run", "state") or {}
        state["training_steps"] = index + 1
        state["training_iterations"] = int(state.get("training_iterations", 0)) + iterations
        store.put("run", "state", state)
        store.put(
            "training_steps",
            f"training-{index:08d}",
            {
                "loss": loss,
                "optimizer_seconds": duration,
                "iterations": iterations,
                "status": "completed",
                "ranks": 2,
            },
        )


def _assembler(run_dir: Path, store: LmdbRunStore) -> RecordedSampleAssembler:
    config = load_config(run_dir)
    return RecordedSampleAssembler(
        run_dir,
        store,
        symbol_dictionary_size=config.model.symbol_dictionary_size,
        discount_rate=config.policy.rewards.discount_rate,
        max_discounted_reward=config.policy.rewards.max_discounted_reward,
        cache_version=config.training.sample_cache_version,
        channels=action_catalog(config.policy),
        recent_action_history=config.model.recent_action_history,
        recent_action_radius=config.training.recent_action_image_radius,
        recent_action_decay=config.training.recent_action_image_decay,
        image_downscale_ratio=config.model.image_downscale_ratio,
        crop_size=(config.training.crop_width, config.training.crop_height),
        next_crop_size=(config.training.next_crop_width, config.training.next_crop_height),
        crop_random=(config.training.crop_random_x, config.training.crop_random_y),
        source=random.Random(config.seed),
    )


def _store(run_dir: Path, readonly: bool = False) -> LmdbRunStore:
    config = load_config(run_dir)
    return LmdbRunStore(
        run_dir / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
        compression_level=config.storage.codec_compression_level,
        readonly=readonly,
    )


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])
