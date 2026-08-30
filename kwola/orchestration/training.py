"""One explicit optimizer runner with rank-zero checkpoint publication."""

import copy
import time
from pathlib import Path
from typing import Any

import torch

from kwola.agent import TraceNet, action_catalog
from kwola.config import load_config
from kwola.hooks import HookRegistry, LifecycleEventName, training_core_hooks
from kwola.storage import (
    LEARNING_SCHEMA_VERSION,
    CheckpointMetadata,
    CheckpointPublisher,
    LmdbRunStore,
    RunManifest,
    load_manifest,
    require_learning_schema,
    verify_checkpoint,
)
from kwola.training.assembler_factory import recorded_sample_assembler
from kwola.training.optimizer import (
    ModelOptimizer,
    OptimizerMetrics,
    summarize_optimizer_metrics,
)
from kwola.training.replay import ReplaySampler, require_replay_budget
from kwola.training.samples import RecordedSampleAssembler

from .lifecycle import RunnerLifecycle
from .results import RunnerResult


class TrainingRunner:
    def __init__(self, run_dir: Path, hooks: HookRegistry | None = None) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)
        self._hooks = hooks or HookRegistry(training_core_hooks(run_dir, self._config))
        self._lifecycle = RunnerLifecycle(self._hooks, run_dir.name, time.time)

    def run(self, gpu: int | None = None) -> RunnerResult:
        result: RunnerResult
        primary_error: BaseException | None = None
        try:
            self._dispatch(LifecycleEventName.RUN_STARTED)
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
        except BaseException as error:
            primary_error = error
            raise
        finally:
            self._lifecycle.finish(primary_error)
        return result.model_copy(update={"warnings": self._lifecycle.warnings})

    def _run_single(self, gpu: int | None) -> RunnerResult:
        started = time.perf_counter()
        device = torch.device("cpu" if gpu is None else f"cuda:{gpu}")
        if gpu is not None:
            torch.cuda.set_device(gpu)
        with self._store() as store:
            (
                step_index,
                trace_count,
                trained_trace_count,
                training_index,
                requested_iterations,
                sample_credit,
            ) = self._state(store)
            if trace_count == 0:
                raise RuntimeError("training requires at least one recorded browser trace")
            self._assembler(store).prepare_cache(self._config.training.sample_cache_workers)
        budget = require_replay_budget(
            trace_count - trained_trace_count,
            requested_iterations,
            self._config.training.batch_size,
            1,
            self._config.training.replay_samples_per_new_trace,
            sample_credit,
            trace_count,
        )
        iteration_count = budget.iterations
        step_id = f"training-{step_index:08d}"
        model = TraceNet(
            self._config.model, num_actions=len(action_catalog(self._config.policy))
        ).to(device)
        optimizer = ModelOptimizer(model, self._config.training)
        manifest = load_manifest(self._run_dir)
        target_model = copy.deepcopy(model).to(device).eval()
        self._load_checkpoint(model, target_model, optimizer, manifest.checkpoint)
        metrics = self._iterations(
            optimizer,
            model,
            target_model,
            device,
            step_index,
            training_index,
            iteration_count,
        )
        checkpoint = self._maybe_publish(model, target_model, optimizer, manifest, step_index + 1)
        self._record(
            step_id,
            metrics,
            iteration_count,
            trace_count,
            budget.remaining_sample_credit,
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
                "present_loss": metrics.present_loss,
                "future_loss": metrics.future_loss,
                "mean_selected_q": metrics.mean_selected_q,
                "mean_bootstrap_target": metrics.mean_bootstrap_target,
                "mean_absolute_td_error": metrics.mean_absolute_td_error,
                "gradient_norm": metrics.gradient_norm,
                "conservative_q_loss": metrics.conservative_q_loss,
            },
        )

    def _iterations(
        self,
        optimizer: ModelOptimizer,
        model: TraceNet,
        target: TraceNet,
        device: torch.device,
        training_step: int,
        training_index: int,
        iteration_count: int,
    ) -> OptimizerMetrics:
        results = []
        count = iteration_count
        with self._store() as store:
            assembler = self._assembler(
                store, freeze_records=True, random_seed=self._config.seed + training_step
            )
            sampler = ReplaySampler(
                assembler.trace_count(),
                self._config.training.batch_size,
                1,
                0,
                self._config.seed,
                training_step,
            )
            for iteration in range(count):
                batch = assembler.assemble(
                    batch_size=self._config.training.batch_size,
                    device=device,
                    impossible_reward=self._config.policy.rewards.impossible_action,
                    sample_indexes=sampler.batch_indexes(iteration),
                )
                index = training_index + iteration
                results.append(
                    optimizer.step_training(
                        batch,
                        target,
                        self._config.policy.rewards.discount_rate,
                        self._config.policy.rewards.max_discounted_reward,
                    )
                )
                if (index + 1) % self._config.training.target_network_update_every == 0:
                    target.load_state_dict(model.state_dict())
        samples = count * self._config.training.batch_size
        return summarize_optimizer_metrics(results, samples)

    def _assembler(
        self,
        store: LmdbRunStore,
        *,
        freeze_records: bool = False,
        random_seed: int | None = None,
    ) -> RecordedSampleAssembler:
        return recorded_sample_assembler(
            self._run_dir,
            store,
            self._config,
            freeze_records=freeze_records,
            random_seed=random_seed,
        )

    def _maybe_publish(
        self,
        model: TraceNet,
        target_model: TraceNet,
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
                {
                    "learning_schema_version": LEARNING_SCHEMA_VERSION,
                    "model": model.state_dict(),
                    "target_model": target_model.state_dict(),
                    "optimizer": optimizer.optimizer.state_dict(),
                },
                stream,
            ),
            manifest=manifest,
            now=time.time(),
        )
        assert published is not None
        checkpoint, _updated_manifest = published
        return checkpoint

    def _load_checkpoint(
        self,
        model: TraceNet,
        target_model: TraceNet,
        optimizer: ModelOptimizer,
        metadata: CheckpointMetadata | None,
    ) -> None:
        if metadata is None:
            return
        device = next(model.parameters()).device
        checkpoint = verify_checkpoint(self._run_dir, metadata)
        payload = require_learning_schema(
            torch.load(checkpoint, map_location=device, weights_only=True)
        )
        model.load_state_dict(payload["model"], strict=True)  # type: ignore[arg-type]
        target_model.load_state_dict(payload["target_model"], strict=True)  # type: ignore[arg-type]
        optimizer.optimizer.load_state_dict(payload["optimizer"])  # type: ignore[arg-type]

    def _state(self, store: LmdbRunStore) -> tuple[int, int, int, int, int, int]:
        state = store.get("run", "state") or {}
        return (
            int(state.get("training_steps", 0)),
            sum(1 for _ in store.scan("traces")),
            int(state.get("training_trace_count", 0)),
            int(state.get("training_iterations", 0)),
            int(
                state.get(
                    "scheduled_training_iterations",
                    self._config.training.batches_per_iteration,
                )
            ),
            int(state.get("replay_sample_credit", 0)),
        )

    def _record(
        self,
        step_id: str,
        metrics: OptimizerMetrics,
        iterations: int,
        trace_count: int,
        replay_sample_credit: int = 0,
    ) -> None:
        with self._store() as store:

            def complete(current: dict[str, Any] | None) -> dict[str, Any]:
                state = dict(current or {})
                state["training_steps"] = int(state.get("training_steps", 0)) + 1
                state["training_iterations"] = int(state.get("training_iterations", 0)) + iterations
                state["training_trace_count"] = trace_count
                state["replay_sample_credit"] = replay_sample_credit
                return state

            store.update("run", "state", complete)
            store.put(
                "training_steps",
                step_id,
                {
                    "loss": metrics.loss,
                    "optimizer_seconds": metrics.duration_seconds,
                    "samples_per_second": metrics.samples_per_second,
                    "present_loss": metrics.present_loss,
                    "future_loss": metrics.future_loss,
                    "mean_selected_q": metrics.mean_selected_q,
                    "mean_bootstrap_target": metrics.mean_bootstrap_target,
                    "mean_absolute_td_error": metrics.mean_absolute_td_error,
                    "gradient_norm": metrics.gradient_norm,
                    "conservative_q_loss": metrics.conservative_q_loss,
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
        self._lifecycle.dispatch(name, subject_id, payload)
