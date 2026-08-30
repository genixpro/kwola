"""LMDB-backed indexed records."""

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Self

import lmdb  # type: ignore[import-untyped]

from .codec import BinaryCodec, CodecError


class RecordCorruptionError(RuntimeError):
    pass


class StorageFullError(RuntimeError):
    pass


class LmdbRunStore:
    def __init__(
        self,
        path: Path,
        *,
        map_size: int = 4 * 1024**3,
        compression_level: int = 3,
        readonly: bool = False,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._environment = lmdb.open(
            str(path),
            map_size=map_size,
            readonly=readonly,
            create=not readonly,
            subdir=True,
            lock=True,
            sync=True,
            metasync=True,
            readahead=readonly,
        )
        self._codec = BinaryCodec(compression_level)
        self._readonly = readonly

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._environment.close()

    @property
    def readonly(self) -> bool:
        return self._readonly

    def put(self, collection: str, key: str, value: Mapping[str, Any]) -> None:
        if self._readonly:
            raise PermissionError("run store is read-only")
        try:
            with self._environment.begin(write=True) as transaction:
                transaction.put(
                    self._key(collection, key),
                    self._codec.encode(value),
                    overwrite=True,
                )
        except lmdb.MapFullError as error:
            raise StorageFullError("run database has exhausted its configured map size") from error

    def get(self, collection: str, key: str) -> dict[str, Any] | None:
        with self._environment.begin() as transaction:
            payload = transaction.get(self._key(collection, key))
        if payload is None:
            return None
        try:
            return self._codec.decode(bytes(payload))
        except CodecError as error:
            raise RecordCorruptionError(f"corrupt {collection} record {key}: {error}") from error

    def update(
        self,
        collection: str,
        key: str,
        transform: Callable[[dict[str, Any] | None], Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Atomically read, transform, and replace one record across processes."""
        if self._readonly:
            raise PermissionError("run store is read-only")
        encoded_key = self._key(collection, key)
        try:
            with self._environment.begin(write=True) as transaction:
                payload = transaction.get(encoded_key)
                current = self._codec.decode(bytes(payload)) if payload is not None else None
                updated = dict(transform(current))
                transaction.put(encoded_key, self._codec.encode(updated), overwrite=True)
                return updated
        except lmdb.MapFullError as error:
            raise StorageFullError("run database has exhausted its configured map size") from error
        except CodecError as error:
            raise RecordCorruptionError(f"corrupt {collection} record {key}: {error}") from error

    def delete(self, collection: str, key: str) -> bool:
        if self._readonly:
            raise PermissionError("run store is read-only")
        with self._environment.begin(write=True) as transaction:
            return bool(transaction.delete(self._key(collection, key)))

    def scan(self, collection: str) -> Iterator[tuple[str, dict[str, Any]]]:
        yield from self.scan_prefix(collection, "")

    def scan_prefix(self, collection: str, key_prefix: str) -> Iterator[tuple[str, dict[str, Any]]]:
        collection_prefix = self._prefix(collection)
        if "\0" in key_prefix:
            raise ValueError("invalid record key prefix")
        prefix = collection_prefix + key_prefix.encode("utf-8")
        with self._environment.begin() as transaction:
            cursor = transaction.cursor()
            if not cursor.set_range(prefix):
                return
            for raw_key, payload in cursor:
                if not raw_key.startswith(prefix):
                    break
                key = raw_key[len(collection_prefix) :].decode("utf-8")
                try:
                    yield key, self._codec.decode(bytes(payload))
                except CodecError as error:
                    message = f"corrupt {collection} record {key}: {error}"
                    raise RecordCorruptionError(message) from error

    @staticmethod
    def _prefix(collection: str) -> bytes:
        if not collection or "\0" in collection:
            raise ValueError("invalid collection name")
        return collection.encode("utf-8") + b"\0"

    @classmethod
    def _key(cls, collection: str, key: str) -> bytes:
        if not key or "\0" in key:
            raise ValueError("invalid record key")
        return cls._prefix(collection) + key.encode("utf-8")
