"""One explicit optimizer runner with rank-zero checkpoint publication."""

import copy
import time
from pathlib import Path

import torch

from kwola.agent import TraceNet
from kwola.config import load_config
from kwola.storage import CheckpointPublisher, LmdbRunStore, RunManifest, load_manifest
from kwola.training.optimizer import ModelOptimizer
from kwola.training.samples import RecordedSampleAssembler, TrainingBatch

from .results import RunnerResult


class TrainingRunner:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)

    def run(self, gpu: int | None = None) -> RunnerResult:
        if gpu is None and self._config.training.world_size > 1:
            from kwola.training.distributed import run_distributed_training

            return run_distributed_training(self._run_dir)
        return self._run_single(gpu)

    def _run_single(self, gpu: int | None) -> RunnerResult:
        started = time.perf_counter()
        device = torch.device("cpu" if gpu is None else f"cuda:{gpu}")
        if gpu is not None:
            torch.cuda.set_device(gpu)
        with self._store() as store:
            step_index, trace_count = self._state(store)
            if trace_count == 0:
                raise RuntimeError("training requires at least one recorded browser trace")
            batch = self._batch(store, device)
        step_id = f"training-{step_index:08d}"
        model = TraceNet(self._config.model, num_actions=6).to(device)
        optimizer = ModelOptimizer(model, self._config.training)
        manifest = load_manifest(self._run_dir)
        checkpoint_file = manifest.checkpoint.file if manifest.checkpoint else None
        self._load_checkpoint(model, optimizer, checkpoint_file)
        target_model = copy.deepcopy(model).to(device).eval()
        metrics = optimizer.step_training(
            batch,
            target_model,
            step_index,
            self._config.policy.rewards.discount_rate,
            self._config.policy.rewards.max_discounted_reward,
        )
        checkpoint = self._publish(model, optimizer, manifest, step_index + 1)
        self._record(step_id, metrics.loss, metrics.duration_seconds)
        return RunnerResult(
            status="completed",
            step_id=step_id,
            duration_seconds=time.perf_counter() - started,
            artifacts=(str(checkpoint.relative_to(self._run_dir)),),
            metrics={
                "loss": metrics.loss,
                "optimizer_seconds": metrics.duration_seconds,
                "samples_per_second": metrics.samples_per_second,
            },
        )

    def _batch(
        self, store: LmdbRunStore, device: torch.device
    ) -> TrainingBatch:
        assembler = RecordedSampleAssembler(
            self._run_dir,
            store,
            symbol_dictionary_size=self._config.model.symbol_dictionary_size,
            discount_rate=self._config.policy.rewards.discount_rate,
            max_discounted_reward=self._config.policy.rewards.max_discounted_reward,
            cache_version=self._config.training.sample_cache_version,
        )
        return assembler.assemble(
            batch_size=self._config.training.batch_size,
            edge=64 if self._config.profile == "testing" else 320,
            device=device,
            impossible_reward=self._config.policy.rewards.impossible_action,
        )

    def _publish(
        self,
        model: TraceNet,
        optimizer: ModelOptimizer,
        manifest: RunManifest,
        generation: int,
    ) -> Path:
        publisher = CheckpointPublisher(
            self._run_dir, self._config.storage.checkpoints_directory
        )
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

    @staticmethod
    def _state(store: LmdbRunStore) -> tuple[int, int]:
        state = store.get("run", "state") or {}
        return int(state.get("training_steps", 0)), sum(1 for _ in store.scan("traces"))

    def _record(self, step_id: str, loss: float, duration: float) -> None:
        with self._store() as store:
            state = store.get("run", "state") or {}
            state["training_steps"] = int(state.get("training_steps", 0)) + 1
            store.put("run", "state", state)
            store.put(
                "training_steps",
                step_id,
                {"loss": loss, "optimizer_seconds": duration, "status": "completed"},
            )

    def _store(self) -> LmdbRunStore:
        return LmdbRunStore(
            self._run_dir / self._config.storage.database_directory,
            map_size=self._config.storage.database_map_size_bytes,
            compression_level=self._config.storage.codec_compression_level,
        )
