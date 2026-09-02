"""Portable paths for the CAViAR repository.

Environment variables:
  CAVIAR_VIDEO_ROOT   Directory containing CCD/Nexar mp4 files
  CAVIAR_DATA_ROOT    Directory containing annotation JSON (default: <repo>/data)
  CAVIAR_RESULTS_DIR  Directory for prediction / eval outputs (default: <repo>/results)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("CAVIAR_DATA_ROOT", REPO_ROOT / "data")).expanduser().resolve()


def video_root() -> Path:
    return Path(os.environ.get("CAVIAR_VIDEO_ROOT", REPO_ROOT / "data" / "videos")).expanduser().resolve()


def results_dir() -> Path:
    return Path(os.environ.get("CAVIAR_RESULTS_DIR", REPO_ROOT / "results")).expanduser().resolve()


def train_json() -> Path:
    return data_root() / "train.json"


def test_json() -> Path:
    return data_root() / "test.json"


def full_json() -> Path:
    return data_root() / "final_accident_qa_dataset_multiple.json"


def holdout_dir() -> Path:
    return data_root() / "holdout"


def resolve_video_path(path: str | os.PathLike[str]) -> str:
    """Resolve a dataset video_path to a local file.

    Released annotations store filenames only (e.g. ``nexar_00001.mp4``).
    Absolute or nested paths are reduced to the basename and joined with
    ``CAVIAR_VIDEO_ROOT``. If the original path already exists, it is kept.
    """
    raw = str(path or "")
    if raw and os.path.exists(raw):
        return raw
    name = os.path.basename(raw)
    if not name:
        return raw
    candidate = video_root() / name
    if candidate.exists():
        return str(candidate)
    # Also try the raw relative path under the video root
    nested = video_root() / raw.lstrip("/")
    if nested.exists():
        return str(nested)
    return str(candidate)
