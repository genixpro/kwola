# Kwola architecture

## Component ownership

| Package | Owns | Must not own |
| --- | --- | --- |
| `domain` | Slotted actions, observations, traces, sessions, bugs, and batches | I/O or frameworks |
| `config` | Strict nested Pydantic settings and built-in profiles | Runtime state |
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
  ├─ N continuously resubmitted testing workers
  │   ├─ Playwright Chromium/Firefox browser
  │   └─ context-managed mitmproxy + persistent Babel worker
  ├─ independent training worker
      ├─ rank 0 / CUDA 0 ─┐
      └─ rank 1 / CUDA 1 ─┴─ NCCL DDP
  └─ resource/pipeline telemetry sampler
```

Testing workers immediately receive another browser session when they finish; the trainer has no
barrier with them. Ranks snapshot recorded metadata for a deterministic step-local shard. The parent
prepares one shared-memory CPU batch per rank; each rank then keeps a decoded-image LRU and builds the
next CPU batch on a prefetch thread while the current batch computes. Gradients synchronize through
DDP. After a final barrier, only rank 0 writes the training record and atomically publishes a
checkpoint. Whole traces and image batches never travel through control queues or pickle files.

## Inference and training geometry

The action catalog is stable and sorted: click/clear, explicit custom typing strings, configured
generated typing strategies, optional double/right clicks, and independent up/down scroll channels.
Its order defines both TraceNet output channels and recorded action indexes. Inference and sample
assembly share the legacy screenshot transform: grayscale, aspect-preserving configured downscale,
dimensions rounded upward to a multiple of eight, and values rounded to two decimals. Current-state
training crops are seeded and action-centred; next-state crops use a separately seeded random centre.
Images, action masks, reward masks, coordinates, and recent-action features are cropped together.

The reward and loss equations preserve the historical present/future/state/advantage phases and
auxiliary cursor, execution-feature, and future-symbol heads. Target checkpoints refresh on the
configured global iteration cadence. Exploration combines the action, session, and test-step axes;
repeat-action suppression resets when a previously unseen branch symbol appears.

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
  telemetry/
    pipeline.jsonl
    training-progress.jsonl
```

Indexed records use MessagePack and Zstandard in LMDB. Resource bodies, screenshots, videos, NumPy
data, logs, and checkpoints remain external blobs. Blob and checkpoint writes use a temporary file,
`fsync`, and atomic rename. Prepared-sample cache records include an explicit version and are rebuilt
from traces when absent, stale, or corrupt.

Instrumentation assigns resources a canonical URL identity, hashes bodies into external blobs, and
realigns branch indexes between rewritten versions by branch signatures. The action-map JavaScript
asset is independently versioned and contract-tested against static pages.

## Hooks and failures

Hooks are ordered first by numeric `order`, then by unique name. Each hook declares its subscribed
events and whether it is fatal. A best-effort failure is returned as a structured `HookFailure`; a
fatal failure raises `HookExecutionError` containing the hook, event, error type, and message. Cleanup
runs in reverse order.

Testing's built-in order is telemetry (10), screenshot integrity (20), bug integrity (30), disposable
sample precomputation (40), metrics (50), and best-effort report/video generation (60). Training uses
the metrics hook at order 10. Screenshot and sample integrity failures are fatal; telemetry, bug
audits, metrics, and report generation are best-effort and identify their hook and event on failure.

`WorkerSupervisor` owns worker timeouts, crash detection, log collection, cancellation, graceful join,
forced termination, and queue cleanup. Control and result messages are validated Pydantic objects and
are limited to 1 MiB. DDP owns process-group setup, rank identity, barriers, failure reduction, and
teardown. Browser and proxy lifecycles are context-managed and idempotently closed.

The pipeline telemetry stream records worker submission/completion and periodic host, process-tree,
memory, and NVIDIA GPU samples. Rank zero emits progress during long training steps with separate
assembly, host-to-device transfer, optimizer, memory, and end-to-end rates. These append-only records
remain readable while a run is active through `kwola status`.
