"""
config_loader.py
=================
Centralized configuration loading for StrokeRehabAI.

Design goals
------------
- Single source of truth: all YAML files in `configs/` are merged into
  one config object at startup.
- Dot-access convenience (`cfg.training.batch_size`) in addition to the
  normal dict interface (`cfg["training"]["batch_size"]`).
- A process-wide cached singleton via `get_config()` so modules do not
  need to re-parse YAML on every import.

TODO (next development phase):
- Add environment-variable override support (e.g. STROKEREHAB_GPU__DEVICE_INDEX=1).
- Add schema validation (pydantic or dataclasses) per config section.
- Add CLI override support (--set training.batch_size=32).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent

# The set of YAML files that make up the full application configuration.
DEFAULT_CONFIG_FILES = [
    "camera.yaml",
    "training.yaml",
    "datasets.yaml",
    "model.yaml",
    "dashboard.yaml",
    "visualization.yaml",
    "evaluation.yaml",
    "gpu.yaml",
    "logging.yaml",
    "augmentation.yaml",
    "features.yaml",
    "exercises.yaml",
    "calibration.yaml",
]


class DotDict(dict):
    """A dict subclass that also allows attribute-style access.

    Example:
        cfg = DotDict({"training": {"batch_size": 16}})
        cfg.training.batch_size  # -> 16
        cfg["training"]["batch_size"]  # -> 16
    """

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class ConfigLoader:
    """Loads and merges YAML configuration files into a single object."""

    def __init__(self, config_dir: Optional[Path] = None, files: Optional[list] = None):
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.files = files or DEFAULT_CONFIG_FILES
        self._raw: Dict[str, Any] = {}

    def load(self) -> DotDict:
        """Read every configured YAML file and merge into one dict.

        Returns:
            DotDict: merged configuration, keyed by top-level section
            name (e.g. "camera", "training", ...).
        """
        merged: Dict[str, Any] = {}
        for filename in self.files:
            path = self.config_dir / filename
            if not path.exists():
                logger.warning("Config file not found, skipping: %s", path)
                continue
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            merged = self._deep_merge(merged, data)
        self._raw = merged
        logger.debug("Loaded config sections: %s", list(merged.keys()))
        return DotDict(merged)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge `incoming` into `base`, returning a new dict."""
        result = dict(base)
        for key, value in incoming.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


_cached_config: Optional[DotDict] = None


def load_config(config_dir: Optional[str] = None, force_reload: bool = False) -> DotDict:
    """Load (and cache) the full application configuration.

    Args:
        config_dir: optional override of the configs/ directory.
        force_reload: if True, bypass the cache and reparse all YAML files.
    """
    global _cached_config
    if _cached_config is None or force_reload:
        loader = ConfigLoader(config_dir=Path(config_dir) if config_dir else None)
        _cached_config = loader.load()
    return _cached_config


def get_config() -> DotDict:
    """Return the cached configuration, loading it on first access."""
    if _cached_config is None:
        return load_config()
    return _cached_config


if __name__ == "__main__":
    # Quick manual sanity check: `python -m configs.config_loader`
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    print("Top-level sections:", list(cfg.keys()))
    print("Example access -> training.batch_size:", cfg.training.batch_size)
