import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import msgpack  # type: ignore[import-untyped]
import pytest
import torch
import zstandard

from kwola.config import profile_config
from kwola.storage import (
    AtomicBlobStore,
    BinaryCodec,
    CheckpointIntegrityError,
    CheckpointMetadata,
    CheckpointPublisher,
    CodecError,
    LearningSchemaError,
    LmdbRunStore,
    RecordCorruptionError,
    RunManifest,
    StorageFullError,
    load_manifest,
    require_learning_schema,
    save_manifest,
    verify_checkpoint,
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
        store.put("traces", "session-1-trace-0", {"reward": 1.0})
        store.put("traces", "session-2-trace-0", {"reward": 2.0})
        store.put("sessions", "1", {"status": "running"})
        updated = store.update(
            "sessions",
            "1",
            lambda current: {**(current or {}), "attempts": 1},
        )
        assert updated == {"status": "running", "attempts": 1}
        assert tuple(store.scan_prefix("traces", "session-1-")) == (
            ("session-1-trace-0", {"reward": 1.0}),
        )
        assert store.delete("traces", "missing") is False

    with LmdbRunStore(database, map_size=1024**2, readonly=True) as store:
        assert store.get("traces", "2") == {"reward": 0.5}
        assert tuple(store.scan("traces")) == (
            ("1", {"reward": 0.25}),
            ("2", {"reward": 0.5}),
            ("session-1-trace-0", {"reward": 1.0}),
            ("session-2-trace-0", {"reward": 2.0}),
        )
        with pytest.raises(PermissionError):
            store.put("traces", "3", {})
        with pytest.raises(PermissionError):
            store.update("sessions", "1", lambda current: current or {})


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
    assert verify_checkpoint(tmp_path, updated.checkpoint) == path


def test_loading_a_version_one_manifest_requires_fresh_initialization(tmp_path: Path) -> None:
    config = profile_config("testing", "https://example.com", 42)
    payload = RunManifest.create(
        target=config.target,
        profile=config.profile,
        seed=config.seed,
        enabled_browsers=config.browser.enabled,
    ).model_dump(mode="json")
    payload["schema_version"] = 1
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="initialize a fresh"):
        load_manifest(tmp_path)


def test_schema_v2_checkpoint_round_trips_online_target_and_optimizer(tmp_path: Path) -> None:
    config = profile_config("testing", "https://example.com", 42)
    manifest = RunManifest.create(
        target=config.target,
        profile=config.profile,
        seed=config.seed,
        enabled_browsers=config.browser.enabled,
    )
    save_manifest(manifest, tmp_path)
    payload = {
        "learning_schema_version": 2,
        "model": {"weight": torch.tensor([1.0])},
        "target_model": {"weight": torch.tensor([2.0])},
        "optimizer": {"state": {7: {"step": torch.tensor(3.0)}}},
    }

    published = CheckpointPublisher(tmp_path).publish(
        rank=0,
        generation=1,
        writer=lambda stream: torch.save(payload, stream),
        manifest=manifest,
        now=1.0,
    )

    assert published is not None
    loaded = require_learning_schema(torch.load(published[0], weights_only=True))
    torch.testing.assert_close(loaded["model"]["weight"], torch.tensor([1.0]))  # type: ignore[index]
    torch.testing.assert_close(loaded["target_model"]["weight"], torch.tensor([2.0]))  # type: ignore[index]
    assert loaded["optimizer"]["state"][7]["step"] == 3  # type: ignore[index]


@pytest.mark.parametrize(
    "payload",
    (
        {"learning_schema_version": 1, "model": {}, "target_model": {}, "optimizer": {}},
        {"learning_schema_version": 2, "model": {}, "optimizer": {}},
    ),
)
def test_learning_schema_rejects_legacy_and_incomplete_checkpoints(
    payload: object,
) -> None:
    with pytest.raises(LearningSchemaError, match="fresh"):
        require_learning_schema(payload)


def test_checkpoint_verification_rejects_corruption_and_missing_files(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"valid")
    metadata = CheckpointMetadata(
        generation=1,
        file="checkpoints/model.pt",
        sha256=hashlib.sha256(b"valid").hexdigest(),
        published_at=1.0,
    )
    assert verify_checkpoint(tmp_path, metadata) == checkpoint
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(CheckpointIntegrityError, match="digest mismatch"):
        verify_checkpoint(tmp_path, metadata)
    checkpoint.unlink()
    with pytest.raises(CheckpointIntegrityError, match="missing"):
        verify_checkpoint(tmp_path, metadata)


def test_checkpoint_paths_reject_traversal_absolute_and_symlink_escape(tmp_path: Path) -> None:
    digest = hashlib.sha256(b"outside").hexdigest()
    with pytest.raises(ValueError, match="confined relative"):
        CheckpointMetadata(generation=1, file="../outside.pt", sha256=digest, published_at=1.0)
    with pytest.raises(ValueError, match="confined relative"):
        CheckpointMetadata(generation=1, file="/outside.pt", sha256=digest, published_at=1.0)

    outside = tmp_path.parent / f"{tmp_path.name}-outside.pt"
    outside.write_bytes(b"outside")
    link = tmp_path / "checkpoint.pt"
    link.symlink_to(outside)
    metadata = CheckpointMetadata(
        generation=1, file="checkpoint.pt", sha256=digest, published_at=1.0
    )
    try:
        with pytest.raises(CheckpointIntegrityError, match="escapes"):
            verify_checkpoint(tmp_path, metadata)
    finally:
        outside.unlink()


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
