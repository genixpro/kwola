"""Disposable prepared-sample cache with explicit version rebuilding."""

from collections.abc import Callable, Mapping
from typing import Any

from kwola.storage import LmdbRunStore, RecordCorruptionError


class SampleCache:
    def __init__(self, store: LmdbRunStore, version: int) -> None:
        self._store = store
        self._version = version

    def get_or_rebuild(
        self,
        session_id: str,
        builder: Callable[[], Mapping[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        try:
            cached = self._store.get("sample_cache", session_id)
        except RecordCorruptionError:
            cached = None
        if cached is not None and cached.get("cache_version") == self._version:
            payload = cached.get("payload")
            if isinstance(payload, dict):
                return payload, False
        rebuilt = dict(builder())
        if not self._store.readonly:
            self._store.put(
                "sample_cache",
                session_id,
                {"cache_version": self._version, "payload": rebuilt},
            )
        return rebuilt, True

    def invalidate(self, session_id: str) -> None:
        if not self._store.readonly:
            self._store.delete("sample_cache", session_id)
