"""One explicit optimizer runner with rank-zero checkpoint publication."""

import copy
import random
import time
from pathlib import Path

import torch

from kwola.agent import TraceNet, action_catalog
from kwola.config import load_config
from kwola.hooks import (
    HookRegistry,
    LifecycleEvent,
    LifecycleEventName,
    training_core_hooks,
)
from kwola.storage import CheckpointPublisher, LmdbRunStore, RunManifest, load_manifest
from kwola.training.optimizer import ModelOptimizer, OptimizerMetrics
from kwola.training.samples import RecordedSampleAssembler, TrainingBatch

from .results import RunnerResult


class TrainingRunner:
    def __init__(self, run_dir: Path, hooks: HookRegistry | None = None) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)
        self._hooks = hooks or HookRegistry(training_core_hooks(run_dir, self._config))

    def run(self, gpu: int | None = None) -> RunnerResult:
        self._dispatch(LifecycleEventName.RUN_STARTED)
        try:
            if gpu is None and self._config.training.world_size > 1:
                from kwola.training.distributed import run_distributed_training

                result = run_distributed_training(self._run_dir)
            else:
                result = self._run_single(gpu)
            self._dispatch(
                LifecycleEventName.TRAINING_ITERATION_FINISHED,
                result.step_id,
                tuple(result.metrics.items()),
            )
            return result
        finally:
            self._dispatch(LifecycleEventName.RUN_FINISHED)
            self._hooks.close()

    def _run_single(self, gpu: int | None) -> RunnerResult:
        started = time.perf_counter()
        device = torch.device("cpu" if gpu is None else f"cuda:{gpu}")
        if gpu is not None:
            torch.cuda.set_device(gpu)
        with self._store() as store:
            step_index, trace_count, training_index, iteration_count = self._state(store)
            if trace_count == 0:
                raise RuntimeError("training requires at least one recorded browser trace")
            self._assembler(store).prepare_cache(self._config.training.sample_cache_workers)
        step_id = f"training-{step_index:08d}"
        model = TraceNet(
            self._config.model, num_actions=len(action_catalog(self._config.policy))
        ).to(device)
        optimizer = ModelOptimizer(model, self._config.training)
        manifest = load_manifest(self._run_dir)
        checkpoint_file = manifest.checkpoint.file if manifest.checkpoint else None
        self._load_checkpoint(model, optimizer, checkpoint_file)
        target_model = copy.deepcopy(model).to(device).eval()
        metrics = self._iterations(
            optimizer, model, target_model, device, training_index, iteration_count
        )
        checkpoint = self._maybe_publish(model, optimizer, manifest, step_index + 1)
        self._record(
            step_id,
            metrics.loss,
            metrics.duration_seconds,
            iteration_count,
        )
        return RunnerResult(
            status="completed",
            step_id=step_id,
            duration_seconds=time.perf_counter() - started,
            artifacts=(str(checkpoint.relative_to(self._run_dir)),) if checkpoint else (),
            metrics={
                "loss": metrics.loss,
                "optimizer_seconds": metrics.duration_seconds,
                "samples_per_second": metrics.samples_per_second,
            },
        )

    def _iterations(
        self,
        optimizer: ModelOptimizer,
        model: TraceNet,
        target: TraceNet,
        device: torch.device,
        training_index: int,
        iteration_count: int,
    ) -> OptimizerMetrics:
        results = []
        count = iteration_count
        for iteration in range(count):
            with self._store() as store:
                batch = self._batch(store, device, iteration * self._config.training.batch_size)
            index = training_index + iteration
            results.append(
                optimizer.step_training(
                    batch,
                    target,
                    index,
                    self._config.policy.rewards.discount_rate,
                    self._config.policy.rewards.max_discounted_reward,
                )
            )
            if (index + 1) % self._config.training.target_network_update_every == 0:
                target.load_state_dict(model.state_dict())
        duration = sum(result.duration_seconds for result in results)
        samples = count * self._config.training.batch_size
        return OptimizerMetrics(
            sum(result.loss for result in results) / count,
            duration,
            samples / duration,
        )

    def _batch(self, store: LmdbRunStore, device: torch.device, offset: int = 0) -> TrainingBatch:
        return self._assembler(store).assemble(
            batch_size=self._config.training.batch_size,
            device=device,
            impossible_reward=self._config.policy.rewards.impossible_action,
            offset=offset,
        )

    def _assembler(self, store: LmdbRunStore) -> RecordedSampleAssembler:
        return RecordedSampleAssembler(
            self._run_dir,
            store,
            symbol_dictionary_size=self._config.model.symbol_dictionary_size,
            discount_rate=self._config.policy.rewards.discount_rate,
            max_discounted_reward=self._config.policy.rewards.max_discounted_reward,
            cache_version=self._config.training.sample_cache_version,
            channels=action_catalog(self._config.policy),
            recent_action_history=self._config.model.recent_action_history,
            recent_action_radius=self._config.training.recent_action_image_radius,
            recent_action_decay=self._config.training.recent_action_image_decay,
            image_downscale_ratio=self._config.model.image_downscale_ratio,
            crop_size=(
                self._config.training.crop_width,
                self._config.training.crop_height,
            ),
            next_crop_size=(
                self._config.training.next_crop_width,
                self._config.training.next_crop_height,
            ),
            crop_random=(
                self._config.training.crop_random_x,
                self._config.training.crop_random_y,
            ),
            source=random.Random(self._config.seed),
        )

    def _maybe_publish(
        self,
        model: TraceNet,
        optimizer: ModelOptimizer,
        manifest: RunManifest,
        generation: int,
    ) -> Path | None:
        if generation % self._config.training.checkpoint_every_iterations:
            return None
        publisher = CheckpointPublisher(self._run_dir, self._config.storage.checkpoints_directory)
        published = publisher.publish(
            rank=0,
            generation=generation,
            writer=lambda stream: torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.optimizer.state_dict()}, stream
            ),
            manifest=manifest,
            now=time.time(),
        )
        assert published is not None
        checkpoint, _updated_manifest = published
        return checkpoint

    def _load_checkpoint(
        self, model: TraceNet, optimizer: ModelOptimizer, relative_path: str | None
    ) -> None:
        if relative_path is None:
            return
        device = next(model.parameters()).device
        payload = torch.load(self._run_dir / relative_path, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.optimizer.load_state_dict(payload["optimizer"])

    def _state(self, store: LmdbRunStore) -> tuple[int, int, int, int]:
        state = store.get("run", "state") or {}
        return (
            int(state.get("training_steps", 0)),
            sum(1 for _ in store.scan("traces")),
            int(state.get("training_iterations", 0)),
            int(
                state.get(
                    "scheduled_training_iterations",
                    self._config.training.batches_per_iteration,
                )
            ),
        )

    def _record(self, step_id: str, loss: float, duration: float, iterations: int) -> None:
        with self._store() as store:
            state = store.get("run", "state") or {}
            state["training_steps"] = int(state.get("training_steps", 0)) + 1
            state["training_iterations"] = int(state.get("training_iterations", 0)) + iterations
            store.put("run", "state", state)
            store.put(
                "training_steps",
                step_id,
                {
                    "loss": loss,
                    "optimizer_seconds": duration,
                    "iterations": iterations,
                    "status": "completed",
                },
            )

    def _store(self) -> LmdbRunStore:
        return LmdbRunStore(
            self._run_dir / self._config.storage.database_directory,
            map_size=self._config.storage.database_map_size_bytes,
            compression_level=self._config.storage.codec_compression_level,
        )

    def _dispatch(
        self,
        name: LifecycleEventName,
        subject_id: str | None = None,
        payload: tuple[tuple[str, object], ...] = (),
    ) -> None:
        self._hooks.dispatch(
            LifecycleEvent(
                name=name,
                occurred_at=time.time(),
                run_id=self._run_dir.name,
                subject_id=subject_id,
                payload=payload,
            )
        )
