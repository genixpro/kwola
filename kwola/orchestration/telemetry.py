"""Durable pipeline events and lightweight host/GPU resource sampling."""

import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Self


class TelemetryWriter:
    def __init__(self, run_dir: Path, interval_seconds: float) -> None:
        directory = run_dir / "telemetry"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "pipeline.jsonl"
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.record("pipeline_started", pid=os.getpid())
        self._thread = threading.Thread(target=self._sample_loop, name="telemetry", daemon=True)
        self._thread.start()

    def record(self, event: str, **values: Any) -> None:
        payload = {
            "timestamp": time.time(),
            "monotonic": time.monotonic(),
            "event": event,
            **values,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._stream.write(encoded + "\n")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 2))
            self._thread = None
        self.record("pipeline_stopped")
        self._stream.close()

    def _sample_loop(self) -> None:
        previous = _cpu_totals()
        while not self._stop.wait(self._interval):
            current = _cpu_totals()
            self.record("resources", **_resource_sample(previous, current))
            previous = current


def read_telemetry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _resource_sample(
    previous: tuple[int, int] | None, current: tuple[int, int] | None
) -> dict[str, Any]:
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    sample: dict[str, Any] = {
        "load_1m": load[0],
        "load_5m": load[1],
        "load_15m": load[2],
        "process_max_rss_kib": usage.ru_maxrss,
        "cpu_percent": _cpu_percent(previous, current),
        **_memory_sample(),
        **_process_tree_sample(os.getpid()),
    }
    gpus = _gpu_sample()
    if gpus:
        sample["gpus"] = gpus
    return sample


def _cpu_totals() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
    except (OSError, ValueError, IndexError):
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float:
    if previous is None or current is None:
        return 0.0
    total = current[0] - previous[0]
    idle = current[1] - previous[1]
    return 0.0 if total <= 0 else 100.0 * (total - idle) / total


def _memory_sample() -> dict[str, int]:
    try:
        rows = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return {}
    values = {row.split(":", 1)[0]: int(row.split()[1]) for row in rows if ":" in row}
    return {
        "memory_total_kib": values.get("MemTotal", 0),
        "memory_available_kib": values.get("MemAvailable", 0),
        "swap_used_kib": values.get("SwapTotal", 0) - values.get("SwapFree", 0),
    }


def _process_tree_sample(root: int) -> dict[str, float | int]:
    processes: dict[int, tuple[int, int, int]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            raw = stat_path.read_text()
            pid_text, remainder = raw.split(" ", maxsplit=1)
            fields = remainder[remainder.rfind(")") + 2 :].split()
            processes[int(pid_text)] = (
                int(fields[1]),
                int(fields[11]) + int(fields[12]),
                int(fields[21]),
            )
        except (OSError, ValueError, IndexError):
            continue
    selected = {root}
    changed = True
    while changed:
        children = {pid for pid, values in processes.items() if values[0] in selected}
        changed = not children.issubset(selected)
        selected.update(children)
    ticks = sum(processes.get(pid, (0, 0, 0))[1] for pid in selected)
    rss_pages = sum(processes.get(pid, (0, 0, 0))[2] for pid in selected)
    return {
        "process_count": len(selected),
        "process_cpu_seconds": ticks / max(1, os.sysconf("SC_CLK_TCK")),
        "process_rss_bytes": rss_pages * os.sysconf("SC_PAGE_SIZE"),
    }


def _gpu_sample() -> list[dict[str, float | int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=True)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in result.stdout.splitlines():
        try:
            index, gpu, memory, used, power, temperature = (
                part.strip() for part in line.split(",")
            )
            rows.append(
                {
                    "index": int(index),
                    "gpu_percent": float(gpu),
                    "memory_percent": float(memory),
                    "memory_used_mib": float(used),
                    "power_watts": float(power),
                    "temperature_c": float(temperature),
                }
            )
        except ValueError:
            continue
    return rows
