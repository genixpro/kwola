"""Two-rank recorded-sample training with explicit DDP ownership."""

import copy
import multiprocessing
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
from kwola.storage import (
    LEARNING_SCHEMA_VERSION,
    CheckpointMetadata,
    CheckpointPublisher,
    LmdbRunStore,
    load_manifest,
    require_learning_schema,
    verify_checkpoint,
)

from .assembler_factory import recorded_sample_assembler
from .batch_stream import batches
from .ddp import DistributedCoordinator, DistributedSettings
from .optimizer import ModelOptimizer, OptimizerMetrics, summarize_optimizer_metrics
from .replay import ReplaySampler
from .samples import RecordedSampleAssembler, TrainingBatch
from .spool import batch_to_device, share_batch
from .telemetry import record_training_progress


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
    with _store(run_dir, readonly=True) as store:
        if not any(True for _ in store.scan("traces")):
            raise RuntimeError("training requires at least one recorded browser trace")


def _shared_initial_batches(run_dir: Path) -> tuple[TrainingBatch, ...]:
    config = load_config(run_dir)
    batches = []
    with _store(run_dir, readonly=True) as store:
        state = store.get("run", "state") or {}
        training_step = int(state.get("training_steps", 0))
        assembler = _assembler(run_dir, store, random_seed=config.seed + training_step)
        trace_count = assembler.trace_count()
        for rank in range(config.training.world_size):
            sampler = ReplaySampler(
                trace_count,
                config.training.batch_size,
                config.training.world_size,
                rank,
                config.seed,
                training_step,
            )
            batch = assembler.assemble(
                batch_size=config.training.batch_size,
                device=torch.device("cpu"),
                impossible_reward=config.policy.rewards.impossible_action,
                sample_indexes=sampler.batch_indexes(0),
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
    config = load_config(run_dir)
    if config.training.cpu_threads_per_rank:
        import cv2

        torch.set_num_threads(config.training.cpu_threads_per_rank)
        cv2.setNumThreads(config.training.cpu_threads_per_rank)
    settings = DistributedSettings(rank, world_size, devices[rank], init_method)
    with DistributedCoordinator(settings) as coordinator:
        initial = shared_batches[rank] if shared_batches else None
        (
            metrics,
            model,
            target,
            optimizer,
            step_index,
            iterations,
            assembly_seconds,
            transfer_seconds,
        ) = _rank_step(run_dir, coordinator, initial)
        loss = torch.tensor([metrics.loss], device=coordinator.device)
        distributed.all_reduce(loss, op=distributed.ReduceOp.SUM)
        coordinator.barrier()
        if coordinator.is_publisher:
            result = _publish_result(
                run_dir,
                model,
                target,
                optimizer,
                step_index,
                metrics,
                float(loss.item() / world_size),
                time.perf_counter() - started,
                iterations,
                assembly_seconds,
                transfer_seconds,
            )
            results.put(result.model_dump_json())


def _rank_step(
    run_dir: Path,
    coordinator: DistributedCoordinator,
    initial_batch: TrainingBatch | None = None,
) -> tuple[OptimizerMetrics, TraceNet, TraceNet, ModelOptimizer, int, int, float, float]:
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
    target = copy.deepcopy(model).to(coordinator.device).eval()
    payload = _load_models(run_dir, manifest.checkpoint, model, target, coordinator.device)
    parallel = DistributedDataParallel(
        model,
        device_ids=[coordinator.settings.local_device],
        output_device=coordinator.settings.local_device,
    )
    optimizer = ModelOptimizer(parallel, config.training)
    if payload is not None:
        optimizer.optimizer.load_state_dict(payload["optimizer"])
    metrics, assembly_seconds, transfer_seconds = _rank_iterations(
        run_dir,
        coordinator,
        model,
        target,
        optimizer,
        step_index,
        training_index,
        iteration_count,
        initial_batch,
    )
    return (
        metrics,
        model,
        target,
        optimizer,
        step_index,
        iteration_count,
        assembly_seconds,
        transfer_seconds,
    )


def _rank_iterations(
    run_dir: Path,
    coordinator: DistributedCoordinator,
    model: TraceNet,
    target: TraceNet,
    optimizer: ModelOptimizer,
    training_step: int,
    training_index: int,
    iteration_count: int,
    initial_batch: TrainingBatch | None,
) -> tuple[OptimizerMetrics, float, float]:
    config = load_config(run_dir)
    results = []
    count = iteration_count
    assembly_seconds = 0.0
    transfer_seconds = 0.0
    loop_started = time.perf_counter()
    with _store(run_dir, readonly=True) as store:
        assembler = _assembler(
            run_dir,
            store,
            random_seed=(
                config.seed + training_step * config.training.world_size + coordinator.settings.rank
            ),
        )
        sampler = ReplaySampler(
            assembler.trace_count(),
            config.training.batch_size,
            config.training.world_size,
            coordinator.settings.rank,
            config.seed,
            training_step,
        )
        batch_stream = batches(
            assembler,
            coordinator,
            count,
            initial_batch,
            config.training.batch_size,
            sampler,
            config.policy.rewards.impossible_action,
            config.training.batch_prefetch,
        )
        for iteration, (cpu_batch, assembly_duration) in enumerate(batch_stream):
            assembly_seconds += assembly_duration
            transfer_started = time.perf_counter()
            batch = batch_to_device(cpu_batch, coordinator.device)
            transfer_seconds += time.perf_counter() - transfer_started
            index = training_index + iteration
            iteration_metrics = optimizer.step_training(
                batch,
                target,
                config.policy.rewards.discount_rate,
                config.policy.rewards.max_discounted_reward,
            )
            results.append(iteration_metrics)
            if coordinator.is_publisher and (
                (iteration + 1) % config.training.telemetry_every_iterations == 0
                or iteration + 1 == count
            ):
                _record_progress(
                    run_dir,
                    coordinator,
                    training_index,
                    iteration,
                    count,
                    results,
                    assembly_seconds,
                    transfer_seconds,
                    loop_started,
                )
            if (index + 1) % config.training.target_network_update_every == 0:
                target.load_state_dict(model.state_dict())
    samples = count * config.training.batch_size
    return (
        summarize_optimizer_metrics(results, samples),
        assembly_seconds,
        transfer_seconds,
    )


def _record_progress(
    run_dir: Path,
    coordinator: DistributedCoordinator,
    training_index: int,
    iteration: int,
    count: int,
    results: list[OptimizerMetrics],
    assembly_seconds: float,
    transfer_seconds: float,
    loop_started: float,
) -> None:
    config = load_config(run_dir)
    completed = iteration + 1
    elapsed = time.perf_counter() - loop_started
    global_samples = completed * config.training.batch_size * config.training.world_size
    record_training_progress(
        run_dir,
        event="training_progress",
        rank=coordinator.settings.rank,
        training_iteration=training_index + completed,
        step_iterations_completed=completed,
        step_iterations_total=count,
        end_to_end_seconds=elapsed,
        optimizer_seconds=sum(item.duration_seconds for item in results),
        assembly_seconds=assembly_seconds,
        transfer_seconds=transfer_seconds,
        end_to_end_samples_per_second=global_samples / elapsed,
        present_loss=sum(item.present_loss for item in results) / completed,
        future_loss=sum(item.future_loss for item in results) / completed,
        mean_selected_q=sum(item.mean_selected_q for item in results) / completed,
        mean_bootstrap_target=(sum(item.mean_bootstrap_target for item in results) / completed),
        mean_absolute_td_error=(sum(item.mean_absolute_td_error for item in results) / completed),
        gradient_norm=sum(item.gradient_norm for item in results) / completed,
        gpu_memory_allocated_bytes=torch.cuda.memory_allocated(coordinator.device),
        gpu_memory_reserved_bytes=torch.cuda.memory_reserved(coordinator.device),
    )


def _publish_result(
    run_dir: Path,
    model: TraceNet,
    target: TraceNet,
    optimizer: ModelOptimizer,
    step_index: int,
    metrics: OptimizerMetrics,
    mean_loss: float,
    duration: float,
    iterations: int,
    assembly_seconds: float,
    transfer_seconds: float,
) -> RunnerResult:
    config = load_config(run_dir)
    manifest = load_manifest(run_dir)
    publisher = CheckpointPublisher(run_dir, config.storage.checkpoints_directory)
    published = None
    checkpoint_started = time.perf_counter()
    if (step_index + 1) % config.training.checkpoint_every_iterations == 0:
        published = publisher.publish(
            rank=0,
            generation=step_index + 1,
            writer=lambda stream: torch.save(
                {
                    "learning_schema_version": LEARNING_SCHEMA_VERSION,
                    "model": model.state_dict(),
                    "target_model": target.state_dict(),
                    "optimizer": optimizer.optimizer.state_dict(),
                },
                stream,
            ),
            manifest=manifest,
            now=time.time(),
        )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    _record_step(
        run_dir,
        step_index,
        mean_loss,
        metrics,
        iterations,
        duration,
        assembly_seconds,
        transfer_seconds,
        checkpoint_seconds,
    )
    global_samples = iterations * config.training.batch_size * config.training.world_size
    return RunnerResult(
        status="completed",
        step_id=f"training-{step_index:08d}",
        duration_seconds=duration,
        artifacts=(str(published[0].relative_to(run_dir)),) if published else (),
        metrics={
            "loss": mean_loss,
            "iterations": iterations,
            "optimizer_seconds": metrics.duration_seconds,
            "samples_per_second": metrics.samples_per_second * config.training.world_size,
            "end_to_end_samples_per_second": global_samples / duration,
            "assembly_seconds": assembly_seconds,
            "transfer_seconds": transfer_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "present_loss": metrics.present_loss,
            "future_loss": metrics.future_loss,
            "mean_selected_q": metrics.mean_selected_q,
            "mean_bootstrap_target": metrics.mean_bootstrap_target,
            "mean_absolute_td_error": metrics.mean_absolute_td_error,
            "gradient_norm": metrics.gradient_norm,
        },
    )


def _load_models(
    run_dir: Path,
    checkpoint: CheckpointMetadata | None,
    model: TraceNet,
    target: TraceNet,
    device: torch.device,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    path = verify_checkpoint(run_dir, checkpoint)
    payload = require_learning_schema(torch.load(path, map_location=device, weights_only=True))
    model.load_state_dict(payload["model"], strict=True)  # type: ignore[arg-type]
    target.load_state_dict(payload["target_model"], strict=True)  # type: ignore[arg-type]
    return payload


def _record_step(
    run_dir: Path,
    index: int,
    loss: float,
    metrics: OptimizerMetrics,
    iterations: int,
    end_to_end_seconds: float,
    assembly_seconds: float,
    transfer_seconds: float,
    checkpoint_seconds: float,
) -> None:
    config = load_config(run_dir)
    with _store(run_dir) as store:

        def complete(current: dict[str, Any] | None) -> dict[str, Any]:
            state = dict(current or {})
            state["training_steps"] = index + 1
            state["training_iterations"] = int(state.get("training_iterations", 0)) + iterations
            return state

        store.update("run", "state", complete)
        store.put(
            "training_steps",
            f"training-{index:08d}",
            {
                "loss": loss,
                "optimizer_seconds": metrics.duration_seconds,
                "optimizer_samples_per_second": (
                    metrics.samples_per_second * config.training.world_size
                ),
                "end_to_end_samples_per_second": (
                    iterations
                    * config.training.batch_size
                    * config.training.world_size
                    / end_to_end_seconds
                ),
                "end_to_end_seconds": end_to_end_seconds,
                "assembly_seconds": assembly_seconds,
                "transfer_seconds": transfer_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "present_loss": metrics.present_loss,
                "future_loss": metrics.future_loss,
                "mean_selected_q": metrics.mean_selected_q,
                "mean_bootstrap_target": metrics.mean_bootstrap_target,
                "mean_absolute_td_error": metrics.mean_absolute_td_error,
                "gradient_norm": metrics.gradient_norm,
                "iterations": iterations,
                "status": "completed",
                "ranks": config.training.world_size,
            },
        )


def _assembler(
    run_dir: Path, store: LmdbRunStore, *, random_seed: int | None = None
) -> RecordedSampleAssembler:
    return recorded_sample_assembler(
        run_dir,
        store,
        load_config(run_dir),
        compact_cpu_tensors=True,
        freeze_records=True,
        random_seed=random_seed,
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
