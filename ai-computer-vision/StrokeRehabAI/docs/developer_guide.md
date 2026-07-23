# Developer Guide

## Code layout conventions

- **One responsibility per module.** e.g. `camera/webcam.py` only knows how
  to talk to a physical camera; `camera/camera_manager.py` adds retry/config
  orchestration on top.
- **Config objects, not scattered constants.** New tunables go in the
  relevant `configs/*.yaml` file and are consumed via the merged config
  object (`configs.config_loader.load_config()`), never hardcoded.
- **Torch imports stay function-local where practical**, so modules that
  don't strictly need torch (e.g. `utils/geometry.py`) can be imported/tested
  without it installed. `models/lstm_model.py` is an exception since the
  model class itself must subclass `nn.Module`.
- **Type hints + docstrings on every public function/class.**
- **`TODO` comments mark intentionally deferred work** — grep for `TODO` to
  find every stub left for the next development phase.

## Adding a new model architecture

1. Create `models/<name>_model.py`, subclassing `BaseRehabModel` (and
   `nn.Module`) — follow `models/lstm_model.py` as a template.
2. Register it in `models/model_factory.py`'s `_MODEL_REGISTRY`.
3. Add `architecture: "<name>"`-specific hyperparameters to
   `configs/model.yaml` if needed.
4. Add a `tests/test_<name>_model.py` mirroring `tests/test_model_factory.py`.

## Adding a new dataset

1. Add its entry (`url`, `local_path`, `license_note`) to
   `configs/datasets.yaml` → `datasets.sources`.
2. Implement `datasets/dataset_converter.py:convert_<name>` to produce the
   unified `.npz` format (see `docs/dataset_guide.md`).
3. Add a validator branch in `datasets/dataset_validator.py` if the raw
   format needs bespoke checks.

## Running tests

```powershell
pytest                      # full suite
pytest -k geometry           # single module
pytest --cov=. --cov-report=term-missing   # coverage
```

Tests that require `torch` (e.g. `tests/test_model_factory.py`) use
`pytest.importorskip("torch")` and are skipped automatically if torch isn't
installed in the current environment — useful for fast lint-only CI stages.

## Logging

Use `utils.logger.get_logger(__name__)` in every module — never
`print()` for anything other than CLI-facing user output (see
`main.py:cmd_dashboard` for the one intentional exception). Logging is
configured once, centrally, via `utils/logger.py:configure_logging()`.
