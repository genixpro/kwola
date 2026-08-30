import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from kwola.agent import InferenceDiagnostics
from kwola.config import load_config
from kwola.orchestration.initialize import initialize_run
from kwola.reporting import RichDebugVideoRenderer
from kwola.reporting.service import ReportService, _reward, _trace_index
from kwola.reporting.videos import VideoRenderer
from kwola.storage import LmdbRunStore


def test_report_generates_summary_chart_videos_and_bug_artifact(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 1)
    config = load_config(tmp_path)
    screenshot = tmp_path / "blobs" / "screenshots" / "frame.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(screenshot), np.full((64, 96, 3), 127, dtype=np.uint8))
    trace = {
        "step_id": "testing-00000000",
        "index": 0,
        "reward": 1.25,
        "screenshot": "blobs/screenshots/frame.png",
        "viewport": [96, 64],
        "action": {"kind": "click", "x": 48, "y": 32},
    }
    with LmdbRunStore(
        tmp_path / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
    ) as store:
        store.put("traces", "trace-1", trace)
        store.put("testing_steps", "testing-00000000", {"status": "completed"})
        store.put("training_steps", "training-00000000", {"status": "completed"})
        store.put("bugs", "bug-1", {"fingerprint": "abc123", "message": "boom"})

    artifacts = ReportService(tmp_path).generate()

    summary = json.loads((tmp_path / "reports" / "summary.json").read_text())
    assert summary == {
        "traces": 1,
        "testing_steps": 1,
        "training_steps": 1,
        "bugs": 1,
        "total_reward": 1.25,
    }
    assert tmp_path / "reports" / "rewards.png" in artifacts
    assert tmp_path / "reports" / "bugs" / "abc123.json" in artifacts
    assert (tmp_path / "reports" / "videos" / "testing-00000000-debug.mp4").stat().st_size
    assert (tmp_path / "reports" / "videos" / "testing-00000000-annotated.mp4").stat().st_size


def test_reporting_validation_and_video_timeout(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="numeric"):
        _reward({"reward": "bad"})
    with pytest.raises(TypeError, match="integer"):
        _trace_index({"index": "bad"})
    with pytest.raises(ValueError, match="at least one trace"):
        VideoRenderer(tmp_path).render(tmp_path / "empty.mp4", [], annotated=False)
    with pytest.raises(TimeoutError):
        VideoRenderer._check_deadline(0)
    with pytest.raises(ValueError, match="invalid screenshot"):
        VideoRenderer(tmp_path)._frame(
            {"screenshot": "missing.png", "reward": 0, "action": {}}, annotated=False
        )


def test_scheduled_report_skips_artifacts_that_are_not_due(tmp_path: Path) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 1)
    config_path = tmp_path / "kwola.json"
    payload = json.loads(config_path.read_text())
    payload["reporting"].update(
        charts=True,
        debug_videos=False,
        annotated_videos=False,
        bug_reports=False,
        chart_every_testing_steps=5,
    )
    config_path.write_text(json.dumps(payload))
    config = load_config(tmp_path)
    with LmdbRunStore(
        tmp_path / config.storage.database_directory,
        map_size=config.storage.database_map_size_bytes,
    ) as store:
        store.put("testing_steps", "testing-1", {"status": "completed"})
        store.put("bugs", "bug-1", {"fingerprint": "ignored"})

    artifacts = ReportService(tmp_path).generate(scheduled=True)

    assert artifacts == (tmp_path / "reports" / "summary.json",)


def test_rich_debug_video_composes_frame_and_requests_h264(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize_run("https://example.com", "testing", tmp_path, 2)
    config = load_config(tmp_path)
    screenshot = tmp_path / "blobs" / "screenshots" / "debug.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[:, :48] = (15, 30, 220)
    image[:, 48:] = (220, 80, 15)
    assert cv2.imwrite(str(screenshot), image)
    trace = {
        "step_id": "testing-00000000",
        "index": 0,
        "reward": 1.25,
        "application_fitness_after": 14.0,
        "screenshot_before": "blobs/screenshots/debug.png",
        "viewport": [96, 64],
        "action": {"channel": "click", "source": "model", "x": 48, "y": 32},
    }
    diagnostic = InferenceDiagnostics(
        channel_names=("click",),
        present_rewards=np.full((1, 8, 12), 0.25, dtype=np.float32),
        future_rewards=np.linspace(-1, 1, 96, dtype=np.float32).reshape(1, 8, 12),
        action_masks=np.ones((1, 8, 12), dtype=np.float32),
        recent_actions_image=np.zeros((1, 8, 12), dtype=np.float32),
        recent_actions_vector=np.zeros(1, dtype=np.float32),
        stamp=np.ones((2, 8, 12), dtype=np.float32),
        checkpoint_generation=3,
        predicted_channel="click",
        predicted_x=40,
        predicted_y=30,
        predicted_value=1.5,
        coverage_symbol_count=12,
        recent_symbol_count=4,
    )
    destination = tmp_path / "reports" / "videos" / "rich.mp4"
    commands: list[list[str]] = []

    def transcode(command: list[str], *, check: bool) -> None:
        assert check
        commands.append(command)
        Path(command[-1]).write_bytes(Path(command[6]).read_bytes())

    monkeypatch.setattr("kwola.reporting.debug_video.subprocess.run", transcode)

    RichDebugVideoRenderer(tmp_path, config).render(destination, [trace], [diagnostic])

    capture = cv2.VideoCapture(str(destination))
    decoded, frame = capture.read()
    capture.release()
    assert destination.stat().st_size > 0
    assert decoded
    assert frame.shape == (960, 1920, 3)
    assert not np.all(frame[:, :, 1] > frame[:, :, (0, 2)].max(axis=2))
    assert "libx264" in commands[0]
    assert "yuv420p" in commands[0]
