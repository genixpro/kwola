"""Bounded decoded screenshot cache for recorded-sample assembly."""

from collections import OrderedDict
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .geometry import process_screenshot


class DecodedImageCache:
    def __init__(self, capacity: int, downscale_ratio: float) -> None:
        self._capacity = capacity
        self._downscale_ratio = downscale_ratio
        self._values: OrderedDict[tuple[str, int | None], NDArray[np.float32]] = OrderedDict()

    def decode(self, path: Path, edge: int | None) -> NDArray[np.float32]:
        key = (str(path), edge)
        cached = self._values.get(key)
        if cached is not None:
            self._values.move_to_end(key)
            return cached
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"invalid screenshot blob: {path}")
        decoded_uint8 = cast(NDArray[np.uint8], decoded)
        if edge is None:
            image = process_screenshot(decoded_uint8, self._downscale_ratio)
        else:
            resized = cv2.resize(decoded_uint8, (edge, edge), interpolation=cv2.INTER_AREA)
            image = cast(NDArray[np.float32], resized[:, :, 0].astype(np.float32) / 255.0)
        if self._capacity:
            self._values[key] = image
            self._values.move_to_end(key)
            while len(self._values) > self._capacity:
                self._values.popitem(last=False)
        return image
