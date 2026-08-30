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

- The locked build passed Ruff formatting/linting, strict mypy over 73 source files, 55 local tests,
  the un-excluded architecture limits, and 92.37% branch-aware unit coverage (85% enforced).
- The rig passed 53 unit/architecture tests plus doctor checks for Python, Linux, ffmpeg, LMDB,
  Chromium, Firefox, Torch, two CUDA devices, and NCCL.
- Fresh Firefox/Kros 3 captured six resources, rewrote both JavaScript resources, recorded branch
  activity on all five traces, wrote five screenshots, trained once on GPU 0, atomically published
  checkpoint generation 1, rebuilt one sample-cache record, and generated reward, debug-video, and
  annotated-video artifacts.
- Fresh Chromium/Kros 1 captured 229 resources and rewrote 162 of 200 JavaScript responses; the
  other scripts were branchless or JSONP-like. All five action traces contained branch activity.
- A post-checkpoint Firefox step selected two model actions and three scheduled weighted-random
  actions. A forced-random step recorded only random-policy actions.
- Real two-rank recorded-sample DDP completed while Firefox testing ran concurrently. Rank 0 alone
  advanced the checkpoint generation and LMDB training state. The final process scan found no
  browser, proxy, Babel, or training workers.
- The standard warmed optimizer benchmark passed at 0.087775 seconds median, 273.43 samples/second,
  and 2.727 GiB peak VRAM on CUDA 0. A full two-rank recorded-sample step took 0.790 seconds in the
  measured optimizer section and reported 60.74 samples/second because it includes target-network
  inference and NCCL gradient synchronization; this end-to-end metric is intentionally reported
  separately from the specified warmed benchmark.

Both the baseline and final fresh Kros runs recorded zero bug classes, so there was no discovered
error-class regression to investigate in this acceptance sample.
