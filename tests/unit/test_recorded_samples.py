from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np
import torch

from kwola.storage import AtomicBlobStore, LmdbRunStore
from kwola.training.samples import ACTION_KINDS, RecordedSampleAssembler


def test_recorded_samples_rebuild_from_trace_artifacts(tmp_path: Path) -> None:
    blobs = AtomicBlobStore(tmp_path / "blobs")
    screenshot = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    encoded_ok, encoded = cv2.imencode(".png", screenshot)
    assert encoded_ok
    path = blobs.write("screenshots", "trace.png", encoded.tobytes())
    with LmdbRunStore(tmp_path / "run.lmdb", map_size=1024**2) as store:
        for index in range(2):
            store.put(
                "traces",
                f"trace-{index}",
                _trace(index, str(path.relative_to(tmp_path))),
            )
        assembler = RecordedSampleAssembler(
            tmp_path,
            store,
            symbol_dictionary_size=100,
            discount_rate=0.85,
            max_discounted_reward=10.0,
            cache_version=3,
        )
        batch = assembler.assemble(
            batch_size=2,
            edge=64,
            device=torch.device("cpu"),
            impossible_reward=-10.0,
        )
        assert batch.request.backbone.image.shape == (2, 1, 64, 64)
        assert batch.request.pixel_action_maps.shape == (2, len(ACTION_KINDS), 64, 64)
        assert batch.next_state_valid.tolist() == [True, False]
        assert batch.present_rewards.tolist() == [0.5, 1.5]
        assert store.get("sample_cache", "testing-0") == {
            "cache_version": 3,
            "payload": {"trace_ids": ["trace-0", "trace-1"]},
        }


def _trace(index: int, screenshot: str) -> dict[str, object]:
    return {
        "step_id": "testing-0",
        "index": index,
        "action": {"kind": "click", "x": 32, "y": 32},
        "reward": index + 0.5,
        "branch_symbols": [1, 2 + index],
        "network_symbols": [10],
        "viewport": [64, 64],
        "action_targets": [
            {
                "bounds": [10, 10, 40, 40],
                "click": True,
                "right_click": False,
                "type": False,
                "scroll": False,
                "scroll_up": False,
                "scroll_down": False,
            }
        ],
        "screenshot": screenshot,
    }
