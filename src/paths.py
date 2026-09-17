"""Every location on disk the project reads from or writes to, defined once."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_IMAGES = RAW / "images"
SAMPLE = DATA / "sample"
RESULTS = ROOT / "results"


def image_path(article_id: int) -> Path:
    """Product photo for an article; the Kaggle release shards them by the first three digits."""
    name = f"{article_id:010d}"
    return RAW_IMAGES / name[:3] / f"{name}.jpg"


def embeddings_path(encoder: str) -> Path:
    """Cached image embeddings written by ``src/data/image_embeddings.py``."""
    return DATA / f"image_embeddings_{encoder}.parquet"
