"""Atomic debug and annotated video rendering from trace screenshots."""

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2


class VideoRenderer:
    def __init__(self, run_dir: Path, frames_per_second: float = 2.0) -> None:
        self._run_dir = run_dir
        self._fps = frames_per_second

    def render(
        self,
        path: Path,
        traces: Sequence[Mapping[str, Any]],
        *,
        annotated: bool,
    ) -> Path:
        if not traces:
            raise ValueError("video rendering requires at least one trace")
        frames = [self._frame(trace, annotated) for trace in traces]
        height, width = frames[0].shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".mp4")
        os.close(descriptor)
        temporary = Path(temporary_name)
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            self._fps,
            (width, height),
        )
        if not writer.isOpened():
            temporary.unlink(missing_ok=True)
            raise RuntimeError("OpenCV could not initialize the MP4 video writer")
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        return path

    def _frame(self, trace: Mapping[str, Any], annotated: bool) -> Any:
        screenshot = self._run_dir / str(trace["screenshot"])
        frame = cv2.imread(str(screenshot), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"invalid screenshot: {screenshot}")
        action = trace.get("action", {})
        label = f"{action.get('kind', 'action')} reward={float(trace['reward']):.3f}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if annotated:
            viewport = trace.get("viewport", [frame.shape[1], frame.shape[0]])
            x = int(action.get("x", 0)) * frame.shape[1] // int(viewport[0])
            y = int(action.get("y", 0)) * frame.shape[0] // int(viewport[1])
            cv2.circle(frame, (x, y), 18, (0, 0, 255), 3)
        return frame
