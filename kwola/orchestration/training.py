"""One explicit optimizer runner with rank-zero checkpoint publication."""

import time
from pathlib import Path

import torch

from kwola.agent import TraceNet
from kwola.config import load_config
from kwola.storage import CheckpointPublisher, LmdbRunStore, load_manifest
from kwola.training.batches import diagnostic_batch
from kwola.training.optimizer import ModelOptimizer

from .results import RunnerResult


class TrainingRunner:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._config = load_config(run_dir)

    def run(self, gpu: int | None = None) -> RunnerResult:
        started = time.perf_counter()
        step_index, trace_count = self._state()
        if trace_count == 0:
            raise RuntimeError("training requires at least one recorded browser trace")
        step_id = f"training-{step_index:08d}"
        device = torch.device("cpu" if gpu is None else f"cuda:{gpu}")
        if gpu is not None:
            torch.cuda.set_device(gpu)
        model = TraceNet(self._config.model, num_actions=6).to(device)
        optimizer = ModelOptimizer(model, self._config.training)
        manifest = load_manifest(self._run_dir)
        checkpoint_file = manifest.checkpoint.file if manifest.checkpoint else None
        self._load_checkpoint(model, optimizer, checkpoint_file)
        request = diagnostic_batch(
            batch_size=self._config.training.batch_size,
            num_actions=6,
            edge=64 if self._config.profile == "testing" else 320,
            seed=self._config.seed + step_index,
            device=device,
            impossible_reward=self._config.policy.rewards.impossible_action,
        )
        metrics = optimizer.step(request)
        publisher = CheckpointPublisher(
            self._run_dir, self._config.storage.checkpoints_directory
        )
        generation = step_index + 1
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

    def _load_checkpoint(
        self, model: TraceNet, optimizer: ModelOptimizer, relative_path: str | None
    ) -> None:
        if relative_path is None:
            return
        device = next(model.parameters()).device
        payload = torch.load(self._run_dir / relative_path, map_location=device)
        model.load_state_dict(payload["model"])
        optimizer.optimizer.load_state_dict(payload["optimizer"])

    def _state(self) -> tuple[int, int]:
        with self._store() as store:
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
