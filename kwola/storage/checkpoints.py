"""Rank-zero atomic checkpoint publication."""

import hashlib
import hmac
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, TypeVar

from .manifest import CheckpointMetadata, RunManifest, save_manifest

T = TypeVar("T")
LEARNING_SCHEMA_VERSION = 3


class CheckpointIntegrityError(RuntimeError):
    pass


class LearningSchemaError(RuntimeError):
    pass


def require_learning_schema(payload: object) -> dict[str, object]:
    if (
        not isinstance(payload, dict)
        or payload.get("learning_schema_version") != LEARNING_SCHEMA_VERSION
    ):
        raise LearningSchemaError(
            "legacy learning state is unsupported; initialize a fresh schema-v3 run"
        )
    required = {"model", "target_model", "optimizer"}
    missing = sorted(required - payload.keys())
    if missing:
        raise LearningSchemaError(
            f"invalid schema-v3 checkpoint; missing {', '.join(missing)}; initialize a fresh run"
        )
    return payload


def verify_checkpoint(run_dir: Path, metadata: CheckpointMetadata) -> Path:
    root = run_dir.resolve()
    try:
        checkpoint = (root / metadata.file).resolve(strict=True)
    except OSError as error:
        raise CheckpointIntegrityError(f"checkpoint is missing: {metadata.file}") from error
    if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
        raise CheckpointIntegrityError(f"checkpoint escapes the run directory: {metadata.file}")
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CheckpointIntegrityError(f"checkpoint cannot be read: {metadata.file}") from error
    if not hmac.compare_digest(digest.hexdigest(), metadata.sha256):
        raise CheckpointIntegrityError(f"checkpoint digest mismatch: {metadata.file}")
    return checkpoint


class CheckpointPublisher:
    def __init__(self, run_dir: Path, checkpoint_dir: str = "checkpoints") -> None:
        self.run_dir = run_dir
        self.directory = run_dir / checkpoint_dir
        self.directory.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        *,
        rank: int,
        generation: int,
        writer: Callable[[BinaryIO], T],
        manifest: RunManifest,
        now: float,
    ) -> tuple[Path, RunManifest] | None:
        if rank != 0:
            return None
        final = self.directory / f"checkpoint-{generation:08d}.pt"
        descriptor, temporary_name = tempfile.mkstemp(dir=self.directory, prefix=".checkpoint.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w+b") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            temporary.replace(final)
            metadata = CheckpointMetadata(
                generation=generation,
                file=str(final.relative_to(self.run_dir)),
                sha256=digest,
                published_at=now,
            )
            updated = manifest.model_copy(update={"checkpoint": metadata})
            save_manifest(updated, self.run_dir)
            return final, updated
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
