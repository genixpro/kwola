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
barrier with them. Each training invocation freezes the current trace snapshot and creates a seeded
shuffle from the run seed, training-step index, and replay epoch. DDP ranks consume disjoint slices
of that common permutation. The replay buffer must contain at least one complete configured global
batch. Each newly recorded trace earns `training.replay_samples_per_new_trace` sample credits, eight
by default, while sample indexes come from the complete frozen replay snapshot. Each global update
consumes `batch_size * world_size` credits; partial-batch credit and work deferred by the scheduled
iteration cap are persisted. Consequently an unchanged replay snapshot may continue training only
while previously earned credit remains. The trace-count high-water mark prevents fresh traces from
being credited twice. Explicit single-device training applies the same accounting with world size
one.
The parent prepares one shared-memory CPU batch per rank; each rank then keeps a decoded-image LRU and
builds the next CPU batch on a prefetch thread while the current batch computes. Gradients synchronize
through DDP. After a final barrier, only rank 0 writes the training record and atomically publishes a
checkpoint. Whole traces and image batches never travel through control queues or pickle files.

Each browser slot tracks consecutive worker failures independently. A failed slot retries with
exponential backoff from `browser_retry_base_seconds` up to `browser_retry_max_seconds`; a successful
step resets its counter. The experiment fails on `browser_max_consecutive_failures` (five by default)
and cancels every active supervisor. Training-worker failures remain immediately fatal. Retry,
recovery, terminal-failure, and shutdown events are written to pipeline telemetry.

While a training worker is active, the scheduler collects durations from successful browser steps.
When training completes, its duration is compared with the median browser duration from that exact
window. The next training iteration count moves by `batch_iteration_adjustment` toward the configured
minimum or maximum; a window with no successful browser completion leaves the schedule unchanged.

## Inference and training geometry

The action catalog is stable and sorted: click/clear, explicit custom typing strings, configured
generated typing strategies, optional double/right clicks, and independent up/down scroll channels.
Its order defines both TraceNet output channels and recorded action indexes. Inference and sample
assembly share the legacy screenshot transform: grayscale, aspect-preserving configured downscale,
dimensions rounded upward to a multiple of eight, and values rounded to two decimals. Current-state
training crops are seeded and action-centred. Next-state targets retain the complete processed
viewport and use the same overlapping tile planner and center-weighted reconstruction as inference.
Shape-compatible tiles are evaluated together without padding. Global coordinates, action masks,
action-map availability, reward masks, and recent-action features remain aligned. Convolutional blocks
use mode-independent spatial GroupNorm.

TraceNet retains separate immediate- and discounted-future reward maps. Their masked sum is the
action-value map used directly by greedy inference. Training uses masked Double DQN: the online model
selects the next valid spatial action, the target model evaluates it, terminal transitions receive a
zero future target, and an empty next-action mask also produces zero bootstrap value. Both reward
heads train immediately with Smooth L1 loss. Cursor, execution-feature, and future-symbol auxiliaries
are enabled. Demonstrated regions are pooled; execution and future-symbol heads receive an action
embedding, while cursor remains spatial-only. Execution uses binary cross entropy, cursor uses
categorical cross entropy, and the normalized future-symbol target comes from the target network.
A conservative margin loss lowers the
highest valid action outside the demonstrated region until it is at least the configured margin below
the demonstrated Q value, limiting offline-Q extrapolation without raising the demonstrated value.
Its default weight and margin are both `0.1`; a zero weight disables it. Gradients are clipped and
target checkpoints refresh on the configured global iteration cadence.

Exploration has two independent stages. The scheduled `random` probability first decides whether to
bypass inference and use action-catalog-weighted random behavior. If the model runs, a second draw
against `weighted_random` chooses between Q-weighted sampling over valid model outputs and the greedy
maximum. The probabilities are conditional rather than ordered intervals and either may exceed the
other. Forced-random testing and inference without a checkpoint use action-catalog-weighted random
behavior. Branch novelty is claimed atomically in a campaign-wide LMDB collection before each trace
is committed. Initial-page coverage is claimed without reward. Code and no-code shaping applies only
when branch instrumentation was available; screenshot and URL novelty remain session-local.

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

Run configuration, manifests, and learning checkpoints use learning schema version 3. A checkpoint
contains the online model, target model, and optimizer together. Loading is strict; older runs and
checkpoints are intentionally not migrated and must be replaced by a freshly initialized run.

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
assembly, host-to-device transfer, optimizer, memory, and end-to-end rates. Training results and
stored step records include present, future, and conservative-Q losses; mean selected Q, mean
bootstrap target, mean absolute TD error, and gradient norm. These append-only records remain
readable while a run is active through `kwola status`, including per-slot retry counts, backoff delay,
and the latest worker error.
