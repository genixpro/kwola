import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import msgpack  # type: ignore[import-untyped]
import pytest
import zstandard

from kwola.config import profile_config
from kwola.storage import (
    AtomicBlobStore,
    BinaryCodec,
    CheckpointPublisher,
    CodecError,
    LmdbRunStore,
    RecordCorruptionError,
    RunManifest,
    StorageFullError,
    load_manifest,
    save_manifest,
)


def test_codec_round_trip_and_corruption_detection() -> None:
    codec = BinaryCodec(1)
    payload = codec.encode({"id": "trace-1", "symbols": [1, 2], "raw": b"bytes"})

    assert codec.decode(payload)["symbols"] == [1, 2]
    with pytest.raises(CodecError, match="checksum"):
        codec.decode(payload[:-1] + bytes((payload[-1] ^ 1,)))
    with pytest.raises(CodecError, match="header"):
        codec.decode(b"invalid")
    wrong_version = payload[:4] + b"\x02" + payload[5:]
    with pytest.raises(CodecError, match="version"):
        codec.decode(wrong_version)
    compressed = zstandard.ZstdCompressor().compress(msgpack.packb([1, 2]))
    digest = hashlib.blake2b(compressed, digest_size=16).digest()
    with pytest.raises(CodecError, match="mapping"):
        codec.decode(b"KWDB\x01" + digest + compressed)


def test_lmdb_records_survive_restart_and_scan(tmp_path: Path) -> None:
    database = tmp_path / "run.lmdb"
    with LmdbRunStore(database, map_size=1024**2) as store:
        store.put("traces", "2", {"reward": 0.5})
        store.put("traces", "1", {"reward": 0.25})
        store.put("sessions", "1", {"status": "running"})
        assert store.delete("traces", "missing") is False

    with LmdbRunStore(database, map_size=1024**2, readonly=True) as store:
        assert store.get("traces", "2") == {"reward": 0.5}
        assert tuple(store.scan("traces")) == (
            ("1", {"reward": 0.25}),
            ("2", {"reward": 0.5}),
        )
        with pytest.raises(PermissionError):
            store.put("traces", "3", {})


def test_atomic_blob_store_confines_paths(tmp_path: Path) -> None:
    store = AtomicBlobStore(tmp_path / "blobs")

    path = store.write("screenshots", "session/1.png", b"png")
    assert path.read_bytes() == b"png"
    assert store.list("screenshots") == ("session/1.png",)
    assert store.read("screenshots", "session/1.png") == b"png"
    store.delete("screenshots", "session/1.png")
    assert store.list("screenshots") == ()
    store.delete("screenshots", "missing.png")
    with pytest.raises(ValueError):
        store.write("screenshots", "../escape", b"bad")


def test_only_rank_zero_publishes_checkpoint_and_manifest(tmp_path: Path) -> None:
    config = profile_config("testing", "https://example.com", 42)
    manifest = RunManifest.create(
        target=config.target,
        profile=config.profile,
        seed=config.seed,
        enabled_browsers=config.browser.enabled,
    )
    save_manifest(manifest, tmp_path)
    publisher = CheckpointPublisher(tmp_path)

    assert (
        publisher.publish(
            rank=1,
            generation=1,
            writer=lambda stream: stream.write(b"other-rank"),
            manifest=manifest,
            now=10.0,
        )
        is None
    )
    published = publisher.publish(
        rank=0,
        generation=1,
        writer=lambda stream: stream.write(b"checkpoint"),
        manifest=manifest,
        now=10.0,
    )

    assert published is not None
    path, updated = published
    assert path.read_bytes() == b"checkpoint"
    assert updated.checkpoint is not None
    assert load_manifest(tmp_path) == updated


def test_atomic_writers_remove_temporary_files_on_failure(tmp_path: Path) -> None:
    blobs = AtomicBlobStore(tmp_path / "blobs")
    with patch.object(Path, "replace", side_effect=OSError("disk")):
        with pytest.raises(OSError, match="disk"):
            blobs.write("items", "broken.bin", b"payload")
    assert not tuple((tmp_path / "blobs" / "items").glob(".broken.bin.*"))
    config = profile_config("testing", "https://example.com", 1)
    manifest = RunManifest.create(
        target=config.target,
        profile=config.profile,
        seed=config.seed,
        enabled_browsers=config.browser.enabled,
    )
    publisher = CheckpointPublisher(tmp_path)
    with pytest.raises(RuntimeError, match="writer"):
        publisher.publish(
            rank=0,
            generation=2,
            writer=lambda _stream: (_ for _ in ()).throw(RuntimeError("writer")),
            manifest=manifest,
            now=1.0,
        )
    assert not tuple((tmp_path / "checkpoints").glob(".checkpoint.*"))


def test_lmdb_validates_names_permissions_corruption_and_capacity(tmp_path: Path) -> None:
    database = tmp_path / "small.lmdb"
    with LmdbRunStore(database, map_size=1024**2) as store:
        with pytest.raises(ValueError, match="collection"):
            store.put("", "key", {})
        with pytest.raises(ValueError, match="record key"):
            store.put("items", "", {})
        with store._environment.begin(write=True) as transaction:
            transaction.put(b"items\0corrupt", b"not-a-codec")
        with pytest.raises(RecordCorruptionError, match="corrupt"):
            store.get("items", "corrupt")
        with pytest.raises(RecordCorruptionError, match="corrupt"):
            tuple(store.scan("items"))
        with pytest.raises(StorageFullError):
            store.put("items", "huge", {"payload": os.urandom(2 * 1024**2)})
    with LmdbRunStore(database, readonly=True) as readonly:
        with pytest.raises(PermissionError):
            readonly.delete("items", "key")
