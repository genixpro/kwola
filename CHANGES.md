# Changes

## 1.1.0

- The experiment runner now keeps configured browser environments continuously in flight while a
  single independent trainer cycles, instead of waiting at a test/train barrier. The new default
  `rig` profile uses eight alternating Chromium/Firefox environments and both GPUs, with explicit
  per-process CPU thread budgets to prevent oversubscription on the 32-thread reference host.
- Recorded-sample training now snapshots trace metadata once per training step, caches decoded
  screenshots, assembles batches on CPU, and prefetches the next batch while DDP computes.
  Step-local cache preparation no longer rescans and rereads every historical screenshot.
- Every run records pipeline/resource JSONL telemetry and rank-zero training progress, including
  assembly, transfer, optimizer, checkpoint, GPU-memory, worker, and end-to-end throughput metrics.
  `kwola status RUN_DIR` reports live rates, in-flight workers, and recent CPU/GPU utilization.
- Browser document navigation is now contained to the target's normalized origin whenever
  `browser.prevent_offsite_navigation` is enabled. Add required OAuth or companion origins to
  `browser.allowed_navigation_origins`. Cross-origin APIs and static subresources are unaffected;
  disabling containment preserves unrestricted navigation.
- Checkpoints are verified against their existing manifest SHA-256 digest before every load.
  Absolute, traversing, missing, symlink-escaping, and corrupt checkpoint files are rejected with a
  `CheckpointIntegrityError`, and PyTorch loads use `weights_only=True`.
- Best-effort lifecycle hook failures are emitted through logging and returned as serialized runner
  and worker warnings. Hook shutdown remains reverse ordered, including failed startup/finish paths.
- Runtime dependencies now use Torch 2.13.0 from PyPI, mitmproxy 12.2.3, OpenCV 5, Matplotlib 3.11,
  and the latest Babel 7 CLI/core releases. The custom CUDA 12.6 package source has been removed.
- Time-limited audit exceptions for cryptography, msgpack, and tornado findings constrained by
  mitmproxy 12.2.3 expire on 2026-09-30; CI will fail if they are not removed or renewed.
- Pull requests and pushes now run locked Python 3.12/Node 24 installs, both supported browsers,
  formatting, linting, strict typing, the full branch-coverage suite, dependency policy audits, and
  fresh wheel/sdist smoke installs.
- Browser workers now retry transient failures independently with bounded exponential backoff and
  cancel every active worker after the configured consecutive-failure limit. `kwola status` reports
  retry health and recovery per browser slot.
- Adaptive training now consumes the median duration of successful browser steps completed during
  each training window. Schedule adjustments, retries, recoveries, and terminal failures are durable
  pipeline telemetry events.
- Distributed training records the configured world size instead of assuming two ranks.

### Rollback

Stop active workers, reinstall the 1.0.0 wheel and its lockfiles, and restart against the unchanged
run directory. No configuration or manifest migration is performed by 1.1.0. A 1.0 checkpoint that
fails the new digest verification should be restored from its original correctly published copy,
not bypassed.
