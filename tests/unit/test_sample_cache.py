from pathlib import Path

from kwola.storage import LmdbRunStore
from kwola.training.cache import SampleCache


def test_cache_rebuilds_when_missing_or_version_changes(tmp_path: Path) -> None:
    calls = 0

    def build() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": calls}

    with LmdbRunStore(tmp_path / "database", map_size=1024**2) as store:
        first, first_rebuilt = SampleCache(store, 1).get_or_rebuild("session", build)
        cached, cached_rebuilt = SampleCache(store, 1).get_or_rebuild("session", build)
        updated, updated_rebuilt = SampleCache(store, 2).get_or_rebuild("session", build)

    assert first == cached == {"value": 1}
    assert first_rebuilt is True
    assert cached_rebuilt is False
    assert updated == {"value": 2}
    assert updated_rebuilt is True


def test_readonly_cache_rebuilds_without_publishing(tmp_path: Path) -> None:
    database = tmp_path / "database"
    with LmdbRunStore(database, map_size=1024**2) as store:
        store.put("run", "state", {})
    with LmdbRunStore(database, map_size=1024**2, readonly=True) as store:
        cache = SampleCache(store, 2)

        payload, rebuilt = cache.get_or_rebuild("new-session", lambda: {"trace_ids": ["a"]})
        cache.invalidate("new-session")

        assert rebuilt is True
        assert payload == {"trace_ids": ["a"]}
        assert store.get("sample_cache", "new-session") is None
