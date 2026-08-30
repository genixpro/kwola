"""Isolated HTML and JavaScript response rewriting."""

import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from threading import Lock
from typing import IO, Self


class RewriteError(RuntimeError):
    pass


class HtmlRewriter:
    _integrity = re.compile(
        rb"\s+integrity\s*=\s*(['\"])sha(?:256|384|512)-[A-Za-z0-9+/=]+\1",
        re.IGNORECASE,
    )

    def rewrite(self, source: bytes) -> bytes:
        return self._integrity.sub(b"", source)


class JavaScriptRewriter:
    _branch_counter = re.compile(rb"globalKwolaCounter_\w{8,10}\[1\]\s*\+=\s*1;")

    def __init__(self, repository_root: Path | None = None) -> None:
        self._root = repository_root or Path(__file__).resolve().parents[2]
        self._worker: subprocess.Popen[str] | None = None
        self._lock = Lock()
        self._cache: dict[str, bytes] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        worker = self._worker
        if worker is None:
            return
        if worker.stdin is not None:
            worker.stdin.close()
        try:
            worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            worker.terminate()
            worker.wait(timeout=3)
        self._worker = None

    def rewrite(self, url: str, source: bytes) -> bytes:
        cache_key = hashlib.sha256(url.encode() + b"\0" + source).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        strict, body = _remove_strict_prefix(source.strip())
        resource_id = hashlib.sha256(url.encode()).hexdigest()[:10]
        try:
            output = self._transform(url, body, "script", resource_id)
        except RewriteError as error:
            if "'import' and 'export'" not in str(error):
                raise
            output = self._transform(url, body, "module", resource_id)
        if self._branch_counter.search(output) is None:
            self._cache[cache_key] = source
            return source
        prefix = b'"use strict";\n' if strict else b""
        rewritten = prefix + output
        self._cache[cache_key] = rewritten
        return rewritten

    def _environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        node_path = str(self._root / "node_modules")
        current = environment.get("NODE_PATH")
        environment["NODE_PATH"] = node_path + (os.pathsep + current if current else "")
        return environment

    def _transform(
        self,
        url: str,
        source: bytes,
        source_type: str,
        resource_id: str,
    ) -> bytes:
        with self._lock:
            worker = self._ensure_worker()
            request = {
                "url": url,
                "source": base64.b64encode(source).decode("ascii"),
                "sourceType": source_type,
                "resourceId": resource_id,
            }
            stdin, stdout = _worker_streams(worker)
            stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            stdin.flush()
            response_line = stdout.readline()
        if not response_line:
            raise RewriteError("Babel instrumentation worker exited unexpectedly")
        response = json.loads(response_line)
        if not response.get("ok"):
            raise RewriteError(f"Babel could not instrument {url}: {response.get('error')}")
        return base64.b64decode(str(response["code"]))

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._worker is not None and self._worker.poll() is None:
            return self._worker
        script = Path(__file__).with_name("assets") / "instrument_javascript.cjs"
        if not script.exists() or not (self._root / "node_modules").exists():
            raise RewriteError("pinned Babel dependencies are missing; run npm ci")
        self._worker = subprocess.Popen(
            ["node", str(script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=self._environment(),
        )
        return self._worker


def _remove_strict_prefix(source: bytes) -> tuple[bool, bytes]:
    for prefix in (b"'use strict';", b'"use strict";'):
        if source.startswith(prefix):
            return True, source[len(prefix) :]
    return False, source


def _worker_streams(worker: subprocess.Popen[str]) -> tuple[IO[str], IO[str]]:
    if worker.stdin is None or worker.stdout is None:
        raise RewriteError("Babel instrumentation worker has no communication streams")
    return worker.stdin, worker.stdout
