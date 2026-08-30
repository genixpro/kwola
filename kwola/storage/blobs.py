"""Atomic local blob storage."""

import os
import tempfile
from pathlib import Path, PurePosixPath


class AtomicBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, namespace: str, name: str, data: bytes) -> Path:
        target = self._path(namespace, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target

    def read(self, namespace: str, name: str) -> bytes:
        return self._path(namespace, name).read_bytes()

    def delete(self, namespace: str, name: str) -> None:
        self._path(namespace, name).unlink(missing_ok=True)

    def list(self, namespace: str) -> tuple[str, ...]:
        directory = self._path(namespace, ".")
        if not directory.exists():
            return ()
        return tuple(
            str(path.relative_to(directory))
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        )

    def _path(self, namespace: str, name: str) -> Path:
        relative = PurePosixPath(namespace) / PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("blob paths must remain inside the run directory")
        return self.root.joinpath(*relative.parts)
