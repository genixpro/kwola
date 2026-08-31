"""Multi-rank recorded-sample training with explicit DDP ownership."""

import copy
import multiprocessing
import time
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
from .ddp import available_tcp_port as _free_port
from .distributed_plan import TrainingPlan as _TrainingPlan
from .distributed_plan import raise_open_file_limit as _raise_open_file_limit
from .distributed_plan import training_plan as _training_plan
from .optimizer import (
    ModelOptimizer,
    OptimizerMetrics,
    optimizer_metrics_payload,
    summarize_optimizer_metrics,
)
from .replay import ReplaySampler
from .replay_state import open_replay_store as _store
from .replay_state import require_new_replay as _prepare_cache
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
    _raise_open_file_limit()
    _prepare_cache(run_dir)
    plan = _training_plan(run_dir)
    shared_batches = (
        _shared_initial_batches(run_dir, plan) if config.training.use_shared_memory_spool else ()
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
            plan,
            shared_batches,
        ),
        nprocs=config.training.world_size,
        join=True,
    )
    return RunnerResult.model_validate_json(results.get())


def _shared_initial_batches(run_dir: Path, plan: _TrainingPlan) -> tuple[TrainingBatch, ...]:
    config = load_config(run_dir)
    batches = []
    with _store(run_dir, readonly=True) as store:
        assembler = _assembler(
            run_dir,
            store,
            random_seed=config.seed + plan.step_index,
            trace_ids=plan.trace_ids,
        )
        for rank in range(config.training.world_size):
            sampler = ReplaySampler(
                plan.trace_count,
                config.training.batch_size,
                config.training.world_size,
                rank,
                config.seed,
                plan.step_index,
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
    plan: _TrainingPlan,
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
            trace_count,
            iterations,
            replay_sample_credit,
            assembly_seconds,
            transfer_seconds,
        ) = _rank_step(run_dir, coordinator, plan, initial)
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
                trace_count,
                metrics,
                float(loss.item() / world_size),
                time.perf_counter() - started,
                iterations,
                assembly_seconds,
                transfer_seconds,
                replay_sample_credit,
            )
            results.put(result.model_dump_json())


def _rank_step(
    run_dir: Path,
    coordinator: DistributedCoordinator,
    plan: _TrainingPlan,
    initial_batch: TrainingBatch | None = None,
) -> tuple[OptimizerMetrics, TraceNet, TraceNet, ModelOptimizer, int, int, int, int, float, float]:
    config = load_config(run_dir)
    manifest = load_manifest(run_dir)
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
        plan.step_index,
        plan.training_index,
        plan.iteration_count,
        initial_batch,
        trace_ids=plan.trace_ids,
    )
    return (
        metrics,
        model,
        target,
        optimizer,
        plan.step_index,
        plan.trace_count,
        plan.iteration_count,
        plan.replay_sample_credit,
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
    *,
    trace_ids: tuple[str, ...] | None = None,
) -> tuple[OptimizerMetrics, float, float]:
    config = load_config(run_dir)
    results = []
    assembly_seconds = 0.0
    transfer_seconds = 0.0
    loop_started = time.perf_counter()
    with _store(run_dir, readonly=True) as store:
        seed = _rank_seed(
            config.seed, training_step, config.training.world_size, coordinator.settings.rank
        )
        assembler = _assembler(
            run_dir,
            store,
            random_seed=seed,
            trace_ids=trace_ids,
        )
        replay_size = _snapshot_replay_size(assembler, trace_ids)
        sampler = ReplaySampler(
            replay_size,
            config.training.batch_size,
            config.training.world_size,
            coordinator.settings.rank,
            config.seed,
            training_step,
        )
        batch_stream = batches(
            assembler,
            coordinator,
            iteration_count,
            initial_batch,
            config.training.batch_size,
            sampler,
            config.policy.rewards.impossible_action,
            config.training.batch_prefetch,
            config.training.pin_memory,
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
                or iteration + 1 == iteration_count
            ):
                _record_progress(
                    run_dir,
                    coordinator,
                    training_index,
                    iteration,
                    iteration_count,
                    results,
                    assembly_seconds,
                    transfer_seconds,
                    loop_started,
                )
            if (index + 1) % config.training.target_network_update_every == 0:
                target.load_state_dict(model.state_dict())
    samples = iteration_count * config.training.batch_size
    return summarize_optimizer_metrics(results, samples), assembly_seconds, transfer_seconds


def _rank_seed(seed: int, step: int, world_size: int, rank: int) -> int:
    return seed + step * world_size + rank


def _snapshot_replay_size(
    assembler: RecordedSampleAssembler, trace_ids: tuple[str, ...] | None
) -> int:
    replay_size = assembler.trace_count()
    if trace_ids is not None and replay_size != len(trace_ids):
        raise RuntimeError("distributed ranks must use the complete parent replay snapshot")
    return replay_size


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
    summary = summarize_optimizer_metrics(results, completed * config.training.batch_size)
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
        **optimizer_metrics_payload(summary),
        gpu_memory_allocated_bytes=torch.cuda.memory_allocated(coordinator.device),
        gpu_memory_reserved_bytes=torch.cuda.memory_reserved(coordinator.device),
    )


def _publish_result(
    run_dir: Path,
    model: TraceNet,
    target: TraceNet,
    optimizer: ModelOptimizer,
    step_index: int,
    trace_count: int,
    metrics: OptimizerMetrics,
    mean_loss: float,
    duration: float,
    iterations: int,
    assembly_seconds: float,
    transfer_seconds: float,
    replay_sample_credit: int = 0,
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
        trace_count,
        duration,
        assembly_seconds,
        transfer_seconds,
        checkpoint_seconds,
        replay_sample_credit,
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
            **optimizer_metrics_payload(metrics),
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
    trace_count: int,
    end_to_end_seconds: float,
    assembly_seconds: float,
    transfer_seconds: float,
    checkpoint_seconds: float,
    replay_sample_credit: int = 0,
) -> None:
    config = load_config(run_dir)
    with _store(run_dir) as store:

        def complete(current: dict[str, Any] | None) -> dict[str, Any]:
            state = dict(current or {})
            state["training_steps"] = index + 1
            state["training_iterations"] = int(state.get("training_iterations", 0)) + iterations
            state["training_trace_count"] = trace_count
            state["replay_sample_credit"] = replay_sample_credit
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
                **optimizer_metrics_payload(metrics),
                "iterations": iterations,
                "status": "completed",
                "ranks": config.training.world_size,
            },
        )


def _assembler(
    run_dir: Path,
    store: LmdbRunStore,
    *,
    random_seed: int | None = None,
    trace_ids: tuple[str, ...] | None = None,
) -> RecordedSampleAssembler:
    return recorded_sample_assembler(
        run_dir,
        store,
        load_config(run_dir),
        compact_cpu_tensors=True,
        freeze_records=True,
        trace_ids=trace_ids,
        random_seed=random_seed,
    )
