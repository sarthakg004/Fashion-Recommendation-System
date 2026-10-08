"""Every location on disk the project reads from or writes to, defined once."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW_IMAGES = RAW / "images"
SAMPLE = DATA / "sample"
RESULTS = ROOT / "results"
IMAGE_EMBEDDINGS = DATA / "image_embeddings_fashion_siglip.parquet"
TEXT_EMBEDDINGS = DATA / "text_embeddings_fashion_siglip.parquet"


def image_path(article_id: int) -> Path:
    """Product photo for an article; the Kaggle release shards them by the first three digits."""
    name = f"{article_id:010d}"
    return RAW_IMAGES / name[:3] / f"{name}.jpg"
