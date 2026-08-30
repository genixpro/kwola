"""Rank-zero atomic checkpoint publication."""

import hashlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, TypeVar

from .manifest import CheckpointMetadata, RunManifest, save_manifest

T = TypeVar("T")


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
