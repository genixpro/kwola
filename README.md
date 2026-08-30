# Kwola

Kwola is an autonomous browser-testing and learning system for web applications. Kwola 1.1 uses the
learning schema version 2 and intentionally rejects 1.0 runs and checkpoints; initialize a fresh run
after upgrading. It requires Python 3.12 and targets Linux for production. It supports Playwright
Chromium and Firefox, mitmproxy-based JavaScript branch instrumentation, local LMDB runs, and
PyTorch/NCCL training.

## Install

```sh
uv sync --locked --python 3.12
npm ci
uv run playwright install chromium firefox
uv run kwola doctor
```

Node dependencies are repository-local and pinned. The JavaScript instrumentation service never
depends on a globally installed Babel executable.

When `browser.prevent_offsite_navigation` is enabled, Kwola confines document navigation to the
target's exact origin. OAuth and other required document origins must be listed in
`browser.allowed_navigation_origins`; cross-origin API and static-resource requests remain allowed.
Set `prevent_offsite_navigation` to `false` to retain unrestricted 1.0 behavior.

## Commands

```sh
kwola init URL --profile testing --run-dir RUN_DIR
kwola init URL --run-dir RUN_DIR  # defaults to the throughput-tuned rig profile
kwola run RUN_DIR
kwola test-step RUN_DIR --browser chromium --random
kwola train-step RUN_DIR --gpu 0
kwola report RUN_DIR
kwola doctor --require-gpus 2
kwola benchmark RUN_DIR
kwola status RUN_DIR
kwola proxy install-cert
```

The built-in profiles are `testing`, `standard`, and `rig`. `rig` is the default and continuously
feeds both GPUs from eight parallel browser environments. It budgets two CPU threads per browser
and four per training rank on the 32-thread reference host, uses a measured batch of 48 per GPU,
caches decoded screenshots in RAM, and prefetches CPU batches while the GPUs optimize. `standard`
preserves the lower-throughput two-rank reference configuration. Pass `--gpu INDEX` for an explicit
single-GPU step.

## Learning behavior

TraceNet predicts separate immediate-reward and discounted-future-reward maps over the valid spatial
action catalog. Their sum drives inference. Training uses masked Double DQN, with the online network
selecting the next valid action and the target network evaluating it. Terminal transitions and next
states with no valid action receive no bootstrap value.

Automatic training starts only after the greater of `orchestration.minimum_traces_before_training`
and one configured global batch (`training.batch_size * training.world_size`) of new traces has
arrived. New traces determine the maximum number of duplicate-free updates in that invocation, while
those updates sample from the entire frozen replay snapshot. A successful step persists the
snapshot's trace-count high-water mark, so the same unchanged replay set cannot trigger training
again. An explicit single-GPU `train-step` uses one local batch as its minimum.

Current-state crops remain action-centred with configured jitter. Next-state crops are sampled
randomly, retried until at least one valid action is visible, and fall back to an action-centred crop.
The default conservative-Q term (`training.losses.conservative_q = 0.1`, margin `0.1`) penalizes the
highest valid action outside the demonstrated region when it exceeds the demonstrated Q value minus
the margin; set its weight to `0` to disable it.

Exploration uses two independent decisions. `random` is the probability of bypassing the model for
action-catalog-weighted random behavior. Otherwise the model runs, and `weighted_random` is the
conditional probability of sampling from its valid Q map instead of taking the greedy maximum.
Forced-random steps and runs without a checkpoint use action-catalog-weighted random behavior.

Every run contains strictly validated `kwola.json` and `manifest.json` files, an LMDB database,
content-addressed blobs, disposable prepared-sample cache records, reports, logs, and atomically
published PyTorch checkpoints. Unknown configuration fields are rejected.

Credentials are referenced by environment-variable name and are never written into new run
configuration files. For example, set `browser.autologin.email_environment` and
`browser.autologin.password_environment` in `kwola.json`, then export those named variables before
running Kwola. The equivalent fixed-action fields are `policy.actions.email_environment` and
`policy.actions.password_environment`. Kwola refuses to write new configuration files containing
inline secrets.

See [Architecture](docs/architecture.md) for component and process ownership, storage layout, hook
ordering, and failure behavior. See [Acceptance evidence](docs/acceptance.md) for the recorded
baseline and final rig results.

## Development

```sh
uv run ruff format --check kwola tests
uv run ruff check kwola tests
uv run mypy --strict kwola
uv run pytest
uv run pip-audit --format json --output pip-audit.json
npm audit --json > npm-audit.json
uv run python scripts/audit_dependencies.py --python-json pip-audit.json \
  --npm-json npm-audit.json --exceptions security/advisory-exceptions.json
```

Every high or critical advisory fails this policy unless it has an owned, justified, unexpired
exception, regardless of whether an upstream fix is currently available.

Release acceptance additionally passes `--require-no-exceptions` to the dependency policy and runs
the complete gate from a fresh schema-v2 run on the Linux two-GPU host:

```sh
uv run python scripts/run_rig_acceptance.py \
  --evidence-dir /path/to/new-empty-evidence-directory \
  --kros1-url http://127.0.0.1:3001/ \
  --kros3-url http://127.0.0.1:3003/
```

The runner records command logs, environment versions, metrics, and artifact hashes; verifies both
browsers, instrumentation, checkpoint compatibility/publication, single-GPU and concurrent two-rank
training; enforces the benchmark and zero-exception gates; and rejects leaked runtime processes.
Only update the acceptance evidence document after this complete command passes.

The architecture tests enforce limits of 500 lines per module, 300 lines per class, and 80 lines per
ordinary method. Domain modules cannot import browser, tensor, storage, process, or reporting
infrastructure.

## License

Kwola is licensed under the MIT License. See [LICENSE](LICENSE).
