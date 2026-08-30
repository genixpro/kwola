# Kwola

Kwola is an autonomous browser-testing and learning system for web applications. Kwola 1.1 remains
compatible with 1.0 run configurations and correctly published checkpoints. It requires Python 3.12
and targets Linux for production. It supports Playwright Chromium and Firefox,
mitmproxy-based JavaScript branch instrumentation, local LMDB runs, and PyTorch/NCCL training.

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
and four per training rank on the 32-thread reference host, caches decoded screenshots in RAM, and
prefetches CPU batches while the GPUs optimize. `standard` preserves the conservative two-rank
reference configuration. Pass `--gpu INDEX` for an explicit single-GPU step.

Every run contains strictly validated `kwola.json` and `manifest.json` files, an LMDB database,
content-addressed blobs, disposable prepared-sample cache records, reports, logs, and atomically
published PyTorch checkpoints. Unknown configuration fields are rejected.

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

The architecture tests enforce limits of 500 lines per module, 300 lines per class, and 80 lines per
ordinary method. Domain modules cannot import browser, tensor, storage, process, or reporting
infrastructure.

## License

Kwola is licensed under the MIT License. See [LICENSE](LICENSE).
