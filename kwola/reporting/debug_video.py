"""Legacy-style diagnostic videos for sampled TraceNet browser sessions."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from kwola.agent.diagnostics import InferenceDiagnostics
from kwola.config.models import RunConfig

Frame = NDArray[np.uint8]


class RichDebugVideoRenderer:
    def __init__(self, run_dir: Path, config: RunConfig) -> None:
        self._run_dir = run_dir
        self._config = config

    def render(
        self,
        path: Path,
        traces: Sequence[Mapping[str, Any]],
        diagnostics: Sequence[InferenceDiagnostics | None],
    ) -> Path:
        if not traces or len(traces) != len(diagnostics):
            raise ValueError("debug video requires aligned traces and diagnostics")
        deadline = time.monotonic() + self._config.reporting.video_timeout_seconds
        rewards = np.asarray([float(trace["reward"]) for trace in traces], dtype=np.float32)
        future = _discounted_rewards(rewards, self._config.policy.rewards.discount_rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._temporary_path(path)
        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
            self._config.reporting.debug_video_frames_per_second,
            (1920, 960),
        )
        if not writer.isOpened():
            temporary.unlink(missing_ok=True)
            raise RuntimeError("OpenCV could not initialize the debug video writer")
        try:
            for index, (trace, diagnostic) in enumerate(zip(traces, diagnostics, strict=True)):
                if time.monotonic() > deadline:
                    raise TimeoutError("debug video rendering exceeded its configured timeout")
                writer.write(self._frame(traces, trace, diagnostic, rewards, future, index))
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            writer.release()
        try:
            self._transcode(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _frame(
        self,
        traces: Sequence[Mapping[str, Any]],
        trace: Mapping[str, Any],
        diagnostic: InferenceDiagnostics | None,
        rewards: NDArray[np.float32],
        future: NDArray[np.float32],
        index: int,
    ) -> Frame:
        canvas = np.full((960, 1920, 3), 18, dtype=np.uint8)
        screenshot = cv2.imread(str(self._run_dir / str(trace["screenshot_before"])))
        if screenshot is None:
            raise ValueError(f"invalid screenshot: {trace['screenshot_before']}")
        screenshot = cv2.resize(screenshot, (1280, 720), interpolation=cv2.INTER_AREA)
        canvas[:720, :1280] = screenshot
        self._action_history(canvas, traces, index)
        self._diagnostic_maps(canvas, diagnostic, trace)
        self._timeline(canvas, rewards, future, index)
        self._details(canvas, trace, diagnostic, rewards, future, index)
        return canvas

    @staticmethod
    def _action_history(canvas: Frame, traces: Sequence[Mapping[str, Any]], current: int) -> None:
        start = max(0, current - 5)
        for index in range(start, current):
            age = current - index
            color = (255 // age, 180 // age, 40)
            _circle(canvas, traces[index], color, max(7, 18 - age * 2), 1280, 720)
        _circle(canvas, traces[current], (20, 20, 255), 18, 1280, 720)

    def _diagnostic_maps(
        self,
        canvas: Frame,
        diagnostic: InferenceDiagnostics | None,
        trace: Mapping[str, Any],
    ) -> None:
        if diagnostic is None or diagnostic.present_rewards is None:
            _text(canvas, "No model checkpoint diagnostics", (1320, 80), 0.8, (180, 180, 180))
            return
        assert diagnostic.future_rewards is not None
        channel = _channel_index(trace, diagnostic.channel_names)
        present = diagnostic.present_rewards
        future = diagnostic.future_rewards
        total = present + future
        maps = (
            ("Present: actual", present[channel], cv2.COLORMAP_TURBO),
            ("Future: actual", future[channel], cv2.COLORMAP_TURBO),
            ("Value: actual", total[channel], cv2.COLORMAP_TURBO),
            ("Present: max", present.max(axis=0), cv2.COLORMAP_TURBO),
            ("Future: max", future.max(axis=0), cv2.COLORMAP_TURBO),
            ("Value: max", total.max(axis=0), cv2.COLORMAP_TURBO),
            ("Valid action mask", diagnostic.action_masks[channel], cv2.COLORMAP_BONE),
            (
                "Recent action memory",
                diagnostic.recent_actions_image.max(axis=0),
                cv2.COLORMAP_INFERNO,
            ),
            ("Network stamp", _stamp_map(diagnostic.stamp), cv2.COLORMAP_VIRIDIS),
        )
        for position, (label, values, color_map) in enumerate(maps):
            row, column = divmod(position, 3)
            self._heatmap(canvas, label, values, color_map, 1288 + column * 208, row * 225)

    @staticmethod
    def _heatmap(
        canvas: Frame,
        label: str,
        values: NDArray[np.float32],
        color_map: int,
        left: int,
        top: int,
    ) -> None:
        finite = values[np.isfinite(values)]
        minimum = float(finite.min()) if finite.size else 0.0
        maximum = float(finite.max()) if finite.size else 0.0
        normalized = np.zeros(values.shape, dtype=np.uint8)
        if maximum > minimum:
            normalized = np.clip((values - minimum) / (maximum - minimum) * 255, 0, 255).astype(
                np.uint8
            )
        colored = cv2.applyColorMap(normalized, color_map)
        colored = cv2.resize(colored, (196, 178), interpolation=cv2.INTER_NEAREST)
        canvas[top + 25 : top + 203, left : left + 196] = colored
        _text(canvas, label, (left, top + 17), 0.43, (235, 235, 235))
        _text(canvas, f"{minimum:+.2f}..{maximum:+.2f}", (left, top + 220), 0.38, (170, 170, 170))

    @staticmethod
    def _timeline(
        canvas: Frame,
        rewards: NDArray[np.float32],
        future: NDArray[np.float32],
        current: int,
    ) -> None:
        left, top, width, height = 25, 760, 1000, 170
        cv2.rectangle(canvas, (left, top), (left + width, top + height), (45, 45, 45), 1)
        total = rewards + future
        minimum = min(float(rewards.min()), float(future.min()), float(total.min()), 0.0)
        maximum = max(float(rewards.max()), float(future.max()), float(total.max()), 0.0)
        for values, color, label in (
            (rewards, (80, 220, 80), "present"),
            (future, (40, 170, 255), "future"),
            (total, (255, 180, 60), "total"),
        ):
            points = _graph_points(values, left, top, width, height, minimum, maximum)
            cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
            label_x = left + 10 + 100 * (label != "present") + 90 * (label == "total")
            _text(canvas, label, (label_x, top + 22), 0.5, color)
        x = left + current * width // max(1, len(rewards) - 1)
        cv2.line(canvas, (x, top), (x, top + height), (230, 230, 230), 1)

    def _details(
        self,
        canvas: Frame,
        trace: Mapping[str, Any],
        diagnostic: InferenceDiagnostics | None,
        rewards: NDArray[np.float32],
        future: NDArray[np.float32],
        index: int,
    ) -> None:
        action = trace_action(trace)
        fitness = trace.get("application_fitness_after")
        action_summary = (
            f"Action {action.get('channel')} ({action.get('source')}) "
            f"at {action.get('x')},{action.get('y')}"
        )
        reward_summary = (
            f"Reward present={rewards[index]:+.3f} future={future[index]:+.3f} "
            f"total={rewards[index] + future[index]:+.3f}"
        )
        _text(canvas, f"Frame {index + 1}/{len(rewards)}  {trace['step_id']}", (1050, 780), 0.62)
        _text(canvas, action_summary, (1050, 810), 0.55)
        _text(canvas, reward_summary, (1050, 840), 0.55)
        _text(
            canvas,
            f"Kros fitness={fitness if fitness is not None else 'n/a'} / 104",
            (1050, 870),
            0.55,
        )
        _text(canvas, _reward_signal_text(trace), (1050, 900), 0.45, (185, 220, 185))
        if diagnostic is not None:
            generation = diagnostic.checkpoint_generation or "none"
            prediction = diagnostic.predicted_channel or "none"
            prediction_summary = (
                f"checkpoint={generation} predicted={prediction} "
                f"value={_format(diagnostic.predicted_value)}"
            )
            _text(
                canvas,
                prediction_summary,
                (1050, 930),
                0.46,
            )
            memory_summary = (
                f"memory actions={_recent_action_count(diagnostic)} "
                f"coverage={diagnostic.coverage_symbol_count} "
                f"recent-symbols={diagnostic.recent_symbol_count} "
                f"new-global={len(_campaign_new_branches(trace))}"
            )
            _text(
                canvas,
                memory_summary,
                (25, 950),
                0.43,
                (180, 180, 180),
            )

    @staticmethod
    def _temporary_path(path: Path) -> Path:
        descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".mp4")
        os.close(descriptor)
        return Path(name)

    @staticmethod
    def _transcode(source: Path, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.h264.mp4")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=True,
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def _discounted_rewards(rewards: NDArray[np.float32], rate: float) -> NDArray[np.float32]:
    result = np.zeros_like(rewards)
    current = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        current *= rate
        result[index] = current
        current += float(rewards[index])
    return result


def _graph_points(
    values: NDArray[np.float32],
    left: int,
    top: int,
    width: int,
    height: int,
    minimum: float,
    maximum: float,
) -> NDArray[np.int32]:
    span = max(maximum - minimum, 1e-6)
    return np.asarray(
        [
            (
                left + index * width // max(1, len(values) - 1),
                top + height - int((value - minimum) / span * height),
            )
            for index, value in enumerate(values)
        ],
        dtype=np.int32,
    )


def _circle(
    canvas: Frame,
    trace: Mapping[str, Any],
    color: tuple[int, int, int],
    radius: int,
    width: int,
    height: int,
) -> None:
    action = trace_action(trace)
    viewport = trace.get("viewport", (1920, 1080))
    viewport_width = max(1, int(viewport[0]))
    viewport_height = max(1, int(viewport[1]))
    x = int(action.get("x", 0)) * width // viewport_width
    y = int(action.get("y", 0)) * height // viewport_height
    cv2.circle(canvas, (x, y), radius, color, 3, cv2.LINE_AA)


def _channel_index(trace: Mapping[str, Any], names: tuple[str, ...]) -> int:
    channel = str(trace_action(trace).get("channel", ""))
    return names.index(channel) if channel in names else 0


def _stamp_map(stamp: NDArray[np.float32] | None) -> NDArray[np.float32]:
    if stamp is None:
        return np.zeros((2, 2), dtype=np.float32)
    return stamp.max(axis=0)


def _recent_action_count(diagnostic: InferenceDiagnostics) -> int:
    channels = len(diagnostic.channel_names)
    recent = diagnostic.recent_actions_vector.reshape(-1, channels).sum(axis=1)
    return int(np.count_nonzero(recent))


def _reward_signal_text(trace: Mapping[str, Any]) -> str:
    return "  ".join(
        (
            f"code={'yes' if trace.get('branch_symbols') else 'no'}",
            f"new-code={len(trace.get('new_branch_symbols', []))}",
            f"network={'yes' if trace.get('network_symbols') else 'no'}",
            f"new-network={len(trace.get('new_network_symbols', []))}",
            f"screenshot={'new' if trace.get('screenshot_new') else 'seen'}",
            f"url={'new' if trace.get('url_new') else 'seen'}",
            f"log={'yes' if trace.get('log_output') else 'no'}",
        )
    )


def _campaign_new_branches(trace: Mapping[str, Any]) -> Sequence[Any]:
    value = trace.get("campaign_new_branch_symbols", trace.get("new_branch_symbols", []))
    return value if isinstance(value, Sequence) else ()


def _text(
    image: Frame,
    value: str,
    position: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (235, 235, 235),
) -> None:
    cv2.putText(image, value, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def trace_action(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    action = trace.get("action", {})
    return action if isinstance(action, Mapping) else {}
