# Kwola

Kwola is an autonomous browser-testing and learning system for web applications. This release is a
clean-break architecture for Python 3.12 and Linux. It supports Playwright Chromium and Firefox,
mitmproxy-based JavaScript branch instrumentation, local LMDB runs, and PyTorch/NCCL training.

Existing Kwola run directories, checkpoints, configuration files, and Python APIs are intentionally
unsupported. Create a fresh run.

## Install

```sh
uv sync --locked --python 3.12
npm ci
uv run playwright install chromium firefox
uv run kwola doctor
```

Node dependencies are repository-local and pinned. The JavaScript instrumentation service never
depends on a globally installed Babel executable.

## Commands

```sh
kwola init URL --profile testing --run-dir RUN_DIR
kwola run RUN_DIR
kwola test-step RUN_DIR --browser chromium --random
kwola train-step RUN_DIR --gpu 0
kwola report RUN_DIR
kwola doctor --require-gpus 2
kwola benchmark RUN_DIR
kwola proxy install-cert
```

Only the `testing` and `standard` profiles exist. `standard` uses two NCCL ranks on CUDA devices 0
and 1 when `train-step` is run without `--gpu`. Pass `--gpu INDEX` for an explicit single-GPU step.

Every run contains strictly validated `kwola.json` and `manifest.json` files, an LMDB database,
content-addressed blobs, disposable prepared-sample cache records, reports, logs, and atomically
published PyTorch checkpoints. Unknown configuration fields are rejected.

See [Architecture](docs/architecture.md) for component and process ownership, storage layout, hook
ordering, and failure behavior.

## Development

```sh
uv run ruff format --check kwola tests
uv run ruff check kwola tests
uv run mypy --strict kwola
uv run pytest
```

The architecture tests enforce limits of 500 lines per module, 300 lines per class, and 80 lines per
ordinary method. Domain modules cannot import browser, tensor, storage, process, or reporting
infrastructure.

## License

Kwola is licensed under the MIT License. See [LICENSE](LICENSE).
