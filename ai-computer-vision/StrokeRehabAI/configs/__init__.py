"""Configuration package for StrokeRehabAI.

Exposes a single entry point, `load_config`, which merges all YAML
configuration files under this directory into one nested dictionary
(and dot-accessible object) that the rest of the application consumes.
"""

from .config_loader import ConfigLoader, load_config, get_config

__all__ = ["ConfigLoader", "load_config", "get_config"]
