# Kwola architecture

## Component ownership

| Package | Owns | Must not own |
| --- | --- | --- |
| `domain` | Slotted actions, observations, traces, sessions, bugs, and batches | I/O or frameworks |
| `config` | Strict nested Pydantic settings and the two profiles | Runtime state |
| `browser` | Playwright lifecycle, navigation, action discovery/execution, waits, login, screenshots | Persistence |
| `instrumentation` | Proxy lifecycle, response rewriting, resources, branch and browser telemetry | Test scheduling |
| `agent` | TraceNet, reward and exploration policy | Reports or process management |
| `training` | Recorded-sample assembly/cache, losses, optimization, DDP | Browser control |
| `orchestration` | Testing/training/experiment runners and supervised processes | Learning math |
| `storage` | LMDB records, codecs, blobs, manifests, atomic checkpoints | Domain policy |
| `reporting` | Summaries, charts, videos, and bug artifacts | Agent inference |

Pydantic models are used at configuration, manifest, CLI, and process-message boundaries. Hot-path
state uses frozen or slotted dataclasses and tensors.

## Process topology

```text
kwola CLI
  ├─ testing runner
  │   ├─ Playwright browser
  │   └─ context-managed mitmproxy + persistent Babel worker
  └─ training runner
      ├─ rank 0 / CUDA 0 ─┐
      └─ rank 1 / CUDA 1 ─┴─ NCCL DDP
```

Ranks read recorded artifacts and build deterministic local shards. Gradients synchronize through
DDP. After a final barrier, only rank 0 writes the training record and atomically publishes a
checkpoint. Large images and checkpoints travel by artifact reference, never through process queues.

## Run layout

```text
run/
  kwola.json
  manifest.json
  run.lmdb/
  blobs/
    resources/
    screenshots/
  cache/
  checkpoints/
  reports/
  logs/
```

Indexed records use MessagePack and Zstandard in LMDB. Resource bodies, screenshots, videos, NumPy
data, logs, and checkpoints remain external blobs. Blob and checkpoint writes use a temporary file,
`fsync`, and atomic rename. Prepared-sample cache records include an explicit version and are rebuilt
from traces when absent, stale, or corrupt.

## Hooks and failures

Hooks are ordered first by numeric `order`, then by unique name. Each hook declares its subscribed
events and whether it is fatal. A best-effort failure is returned as a structured `HookFailure`; a
fatal failure raises `HookExecutionError` containing the hook, event, error type, and message. Cleanup
runs in reverse order.

`WorkerSupervisor` owns worker timeouts, crash detection, log collection, cancellation, graceful join,
forced termination, and queue cleanup. Control and result messages are validated Pydantic objects and
are limited to 1 MiB. DDP owns process-group setup, rank identity, barriers, failure reduction, and
teardown. Browser and proxy lifecycles are context-managed and idempotently closed.
