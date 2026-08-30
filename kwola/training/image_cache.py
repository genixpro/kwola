"""Bounded decoded screenshot cache for recorded-sample assembly."""

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from kwola.storage import AtomicBlobStore

from .geometry import process_screenshot


class DecodedImageCache:
    def __init__(
        self,
        capacity: int,
        downscale_ratio: float,
        persistent_directory: Path | None = None,
    ) -> None:
        self._capacity = capacity
        self._downscale_ratio = downscale_ratio
        self._values: OrderedDict[tuple[str, int | None], NDArray[np.float32]] = OrderedDict()
        self._persistent = (
            AtomicBlobStore(persistent_directory / f"v1-ratio-{downscale_ratio:.8f}")
            if persistent_directory is not None
            else None
        )

    def decode(self, path: Path, edge: int | None) -> NDArray[np.float32]:
        key = (str(path), edge)
        cached = self._values.get(key)
        if cached is not None:
            self._values.move_to_end(key)
            return cached
        image = self._persistent_image(path) if edge is None else None
        if image is None:
            image = self._decode_source(path, edge)
            if edge is None:
                self._store_persistent(path, image)
        if self._capacity:
            self._values[key] = image
            self._values.move_to_end(key)
            while len(self._values) > self._capacity:
                self._values.popitem(last=False)
        return image

    def store_encoded(self, path: Path, encoded: bytes) -> None:
        """Populate the persistent training representation while bytes are already hot."""
        if self._persistent is None:
            return
        payload = np.frombuffer(encoded, dtype=np.uint8)
        decoded = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError(f"invalid screenshot blob: {path}")
        image = process_screenshot(cast(NDArray[np.uint8], decoded), self._downscale_ratio)
        self._store_persistent(path, image)

    def _decode_source(self, path: Path, edge: int | None) -> NDArray[np.float32]:
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        # Match live observation encoding and avoid decoding three color channels
        # that are immediately collapsed to grayscale by process_screenshot.
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError(f"invalid screenshot blob: {path}")
        decoded_uint8 = cast(NDArray[np.uint8], decoded)
        if edge is None:
            image = process_screenshot(decoded_uint8, self._downscale_ratio)
        else:
            resized = cv2.resize(decoded_uint8, (edge, edge), interpolation=cv2.INTER_AREA)
            image = cast(NDArray[np.float32], resized.astype(np.float32) / 255.0)
        return image

    def _persistent_image(self, source: Path) -> NDArray[np.float32] | None:
        if self._persistent is None:
            return None
        try:
            encoded = np.frombuffer(
                self._persistent.read("images", self._persistent_name(source)), dtype=np.uint8
            )
        except FileNotFoundError:
            return None
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            return None
        return cast(NDArray[np.float32], decoded.astype(np.float32) / 100.0)

    def _store_persistent(self, source: Path, image: NDArray[np.float32]) -> None:
        if self._persistent is None:
            return
        quantized = np.rint(image * 100).astype(np.uint8)
        encoded_ok, encoded = cv2.imencode(".png", quantized)
        if encoded_ok:
            self._persistent.write("images", self._persistent_name(source), encoded.tobytes())

    @staticmethod
    def _persistent_name(source: Path) -> str:
        return f"{hashlib.sha256(source.name.encode()).hexdigest()}.png"
