# Refactor acceptance evidence

Acceptance host: `therig.local`, Linux 7.0.0-29, Python 3.12.14, two NVIDIA RTX 2070
SUPER GPUs (8 GiB each), CUDA 12.6, NCCL available. The refactor was deployed only to
`/home/bradley/kwola-refactor`. `/home/bradley/kwola` remained present and was not written.

## Baseline snapshot

The installed baseline reported Kwola 0.1.52. Historical `kwola_run_015` targeted Kros 3 with the
standard Chromium profile and contained 12 testing steps, 600 traces, zero bugs, two model files,
12 normal videos, and 12 lossless videos. Its deep-learning model artifact was approximately 210 MB.
Kros 1 and Kros 3 returned HTTP 200 at `127.0.0.1:3001` and `127.0.0.1:3003`.

## Final 1.0.0 results

- The locked build passed Ruff formatting/linting, strict mypy over 80 source files, 73 local tests
  (plus two opt-in rig contracts), all architecture limits, and 91.68% branch-aware coverage (85%
  enforced). Configuration is 98%, rewards 100%, loss math 95%, storage 94–100%, and worker
  supervision 99% covered.
- The rig passed the same suite and doctor checks for Python, Linux, ffmpeg, LMDB, 61.70 GiB of
  `/dev/shm`, Chromium, Firefox, Torch, two CUDA devices, and NCCL. The explicit Kros action contract
  passed in both browsers and exercised click, type, clear, double-click, right-click, up/down scroll,
  action discovery, network waiting, and screenshots.
- Fresh Firefox/Kros 3 captured six resources, rewrote both JavaScript resources, recorded branch and
  network activity on every trace, wrote screenshots, trained ten iterations on GPU 0, atomically
  published checkpoint generation 1, rebuilt disposable cache records, and generated reward,
  debug-video, and annotated-video artifacts. A post-checkpoint step selected two model actions and
  three scheduled weighted-random actions.
- Fresh Chromium/Kros 1 captured 230 resource versions and rewrote 162 JavaScript resources. A second
  run enabled HTML capture and a disposable account; autologin reached the authenticated application
  and all five traces stored both before/after HTML. Navigation remained on the target origin.
- CPU forward/backward/optimizer completed at 0.033262 seconds median. The explicit two-rank NCCL
  diagnostic completed finite optimizer steps on both GPUs. The standard warmed benchmark passed at
  0.087409 seconds median, 274.57 samples/second, and 2.727 GiB peak VRAM on CUDA 0—comfortably inside
  the 1.35 second, 145 samples/second, and 5.0 GiB limits.
- Real standard-profile two-rank recorded-sample DDP completed while Firefox testing ran concurrently.
  The collision test initially exposed a read-only-rank cache publication race; ranks now rebuild
  newly discovered disposable metadata in memory while only the writable precompute owner publishes
  it. The identical rerun passed: two DDP iterations used 1.225 seconds of optimizer time and reported
  78.35 end-to-end samples/second. Rank 0 alone advanced the checkpoint and LMDB training state, only
  one checkpoint file exists, and the final process scan found no browser, proxy, Babel, or training
  workers.

The historical baseline recorded no application bug classes. The fresh Kros 1 run reports two views
of one environmental defect: the application requests `bower_components/chosen/chosen.css`, which
the healthy Kros service independently returns as HTTP 404, and Chromium emits the corresponding
console error. Kros 3 remains at zero bug classes. This is improved network/console capture of a
confirmed missing Kros asset, not a browser-action, instrumentation, or learning regression.

All acceptance runs lived under `/home/bradley/kwola-refactor`. The baseline directory timestamp for
`/home/bradley/kwola` remained `2026-08-29 21:11:20.449445097 -0400` before and after acceptance.
