"""Shared loading helpers for the sampled dataset.

Every model script reads the same three parquet files written by
``data/sample_data.py``, so the loading lives here rather than in five copies.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def load_transactions(split: str | None = None) -> pl.DataFrame:
    """Sampled transactions, optionally one split of ``train`` / ``val`` / ``test``."""
    path = SAMPLE / "transactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run data/sample_data.py first.")
    transactions = pl.read_parquet(path)
    return transactions if split is None else transactions.filter(pl.col("split") == split)


def load_articles() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / "articles.parquet")


def purchases_by_customer(transactions: pl.DataFrame) -> dict[str, list[int]]:
    """``{customer_id: [article_id, ...]}`` in purchase order.

    Used both for ground truth (test split) and for purchase history (train split).
    """
    grouped = transactions.sort("t_dat").group_by("customer_id").agg(pl.col("article_id"))
    return dict(zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list()))
