"""Portable paths for the CAViAR repository.

Environment variables:
  CAVIAR_VIDEO_ROOT   Directory containing CCD/Nexar mp4 files
  CAVIAR_DATA_ROOT    Directory containing annotation JSON (default: <repo>/data)
  CAVIAR_RESULTS_DIR  Directory for prediction / eval outputs (default: <repo>/results)
"""

from __future__ import annotations

import os
from functools import lru_cache
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


def video_id(path: str | os.PathLike[str]) -> str:
    """Numeric Nexar id (e.g. ``00284``) from a dataset ``video_path``.

    Accepts ``00284``, ``00284.mp4``, ``nexar_00284.mp4``, or a longer path.
    """
    name = os.path.basename(str(path or ""))
    stem, _ext = os.path.splitext(name)
    if stem.lower().startswith("nexar_"):
        stem = stem[6:]
    return stem


@lru_cache(maxsize=1)
def _video_index() -> dict[str, Path]:
    """Map numeric id / filename stem → local mp4 under ``CAVIAR_VIDEO_ROOT``."""
    root = video_root()
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for p in root.rglob("*.mp4"):
        vid = video_id(p.name)
        index.setdefault(vid, p)
        index.setdefault(p.stem, p)
        index.setdefault(p.name, p)
    return index


def resolve_video_path(path: str | os.PathLike[str]) -> str:
    """Resolve a dataset video_path to a local file.

    Released Nexar annotations store the numeric clip id only (e.g. ``00284``).
    That id matches ``{id}.mp4`` in `nexar-ai/nexar_collision_prediction
    <https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction>`_
    (train/test-public/test-private subfolders). Absolute paths that already
    exist are kept. Also accepts legacy ``nexar_{id}.mp4`` names.
    """
    raw = str(path or "")
    if raw and os.path.exists(raw):
        return raw

    vid = video_id(raw)
    root = video_root()
    candidates = []
    if vid:
        candidates.extend(
            [
                root / f"{vid}.mp4",
                root / f"nexar_{vid}.mp4",
            ]
        )
    name = os.path.basename(raw)
    if name:
        candidates.append(root / name)
    if raw:
        candidates.append(root / raw.lstrip("/"))

    for cand in candidates:
        if cand.exists():
            return str(cand)

    indexed = _video_index().get(vid) or _video_index().get(name)
    if indexed is not None:
        return str(indexed)

    return str(candidates[0] if candidates else root / raw)
