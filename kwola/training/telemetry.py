"""Low-overhead progress records emitted by the publishing training rank."""

import json
import time
from pathlib import Path
from typing import Any


def record_training_progress(run_dir: Path, **values: Any) -> None:
    directory = run_dir / "telemetry"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), **values}
    with (directory / "training-progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
