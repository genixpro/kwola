"""Content-addressed resource capture backed by the run store."""

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kwola.storage import AtomicBlobStore, LmdbRunStore


class ResourceRegistry:
    def __init__(
        self,
        store: LmdbRunStore,
        blobs: AtomicBlobStore,
        run_dir: Path,
    ) -> None:
        self._store = store
        self._blobs = blobs
        self._run_dir = run_dir

    def capture(
        self,
        *,
        url: str,
        status: int,
        content_type: str,
        headers: Mapping[str, str],
        original: bytes,
        delivered: bytes,
        rewrite_kind: str | None,
    ) -> str:
        content_hash = hashlib.sha256(original).hexdigest()
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        record_id = f"{url_hash}-{content_hash[:16]}"
        suffix = _safe_suffix(content_type)
        blob = self._blobs.write("resources", f"{content_hash}{suffix}", original)
        record: dict[str, Any] = {
            "url": url,
            "status": status,
            "content_type": content_type,
            "headers": dict(headers),
            "content_hash": content_hash,
            "size": len(original),
            "delivered_size": len(delivered),
            "rewrite_kind": rewrite_kind,
            "blob": str(blob.relative_to(self._run_dir)),
        }
        self._store.put("resources", record_id, record)
        self._store.put("resource_urls", url_hash, {"latest": record_id, "url": url})
        return record_id


def _safe_suffix(content_type: str) -> str:
    media_type = content_type.partition(";")[0].strip().lower()
    return {
        "application/javascript": ".js",
        "application/x-javascript": ".js",
        "text/javascript": ".js",
        "text/html": ".html",
        "application/json": ".json",
        "text/css": ".css",
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/svg+xml": ".svg",
    }.get(media_type, ".bin")
