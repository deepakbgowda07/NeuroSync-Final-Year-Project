"""
model_downloader.py
====================
Helper for fetching pretrained model weights (e.g. a released
StrokeRehabAI checkpoint, or MediaPipe model assets) from a remote
URL with resume support and checksum verification.

TODO (next development phase):
- Point DEFAULT_MODEL_REGISTRY at real release URLs once weights exist.
- Add checksum verification (sha256) once the checkpoint is public.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.file_io import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

# Placeholder registry — populate once trained weights are released.
DEFAULT_MODEL_REGISTRY = {
    "rehab_lstm_v1": {
        "url": None,  # TODO: fill in release URL
        "sha256": None,  # TODO: fill in checksum
        "local_path": "weights/checkpoints/rehab_lstm_v1.pt",
    }
}


def download_file(url: str, destination: str, chunk_size: int = 8192) -> Path:
    """Stream-download a file to `destination`, creating parent dirs.

    Uses `requests` if available; otherwise falls back to urllib.
    """
    destination_path = Path(destination)
    ensure_dir(destination_path.parent)

    try:
        import requests

        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(destination_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fh.write(chunk)
    except ImportError:
        import urllib.request

        urllib.request.urlretrieve(url, destination_path)

    logger.info("Downloaded %s -> %s", url, destination_path)
    return destination_path


def get_or_download_model(model_name: str) -> Optional[Path]:
    """Return the local path to a registered model, downloading it if
    a URL is configured and the file is not already present."""
    entry = DEFAULT_MODEL_REGISTRY.get(model_name)
    if entry is None:
        logger.error("Unknown model name in registry: %s", model_name)
        return None

    local_path = Path(entry["local_path"])
    if local_path.exists():
        return local_path

    if not entry.get("url"):
        logger.warning(
            "No download URL configured for '%s' yet. Place weights manually at: %s",
            model_name,
            local_path,
        )
        return None

    return download_file(entry["url"], str(local_path))
