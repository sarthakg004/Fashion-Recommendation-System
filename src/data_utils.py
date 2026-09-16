"""Shared loading helpers for the sampled dataset.

Every model script reads the same three parquet files written by
``data/sample_data.py``, so the loading lives here rather than in five copies.
"""

from __future__ import annotations

import datetime as dt
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


def load_fitting_data() -> pl.DataFrame:
    """Everything a model may learn from before predicting the test week.

    Hyperparameters are chosen on the validation week, and then every model is
    refitted on train *and* val before it predicts test, which is what a weekly
    production retrain does. It matters more than it sounds: fitted on train
    alone, a model's view of the catalog stops a week before the week it is
    predicting, which halves the repurchase signal (2.0% of test purchases are
    repeats of a train article, 3.9% of a train-or-val one) and dates the
    bestseller list (test recall of the last-7-day top 300 goes 0.178 -> 0.223).

    All five models use this, so the comparison stays like for like.
    """
    return load_transactions().filter(pl.col("split") != "test")


def split_at(weeks_back: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fitting frame and evaluation week, counted back from the test week.

    ``weeks_back=0`` reproduces the fixed split exactly: everything before the
    test week to fit on, the test week to score against. Larger values slide
    both windows one week earlier, which is what lets the whole pipeline be run
    against several held-out weeks instead of trusting a single one. A metric
    from one week carries the quirks of that week, and on a fashion catalog
    those are real: a cold snap or a promotion moves the bestseller list enough
    to move the score.
    """
    transactions = load_transactions()
    end = transactions["t_dat"].max() - dt.timedelta(days=7 * weeks_back)
    start = end - dt.timedelta(days=6)
    return (
        transactions.filter(pl.col("t_dat") < start),
        transactions.filter((pl.col("t_dat") >= start) & (pl.col("t_dat") <= end)),
    )


def load_articles() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / "articles.parquet")


def purchases_by_customer(transactions: pl.DataFrame) -> dict[str, list[int]]:
    """``{customer_id: [article_id, ...]}`` in purchase order.

    Used both for ground truth (test split) and for purchase history (train split).
    """
    grouped = transactions.sort("t_dat").group_by("customer_id").agg(pl.col("article_id"))
    return dict(zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list()))
