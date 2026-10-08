"""Shared loading helpers for the sampled dataset.

Every model reads the same parquet files written by ``src/data/sampling.py``, so
the loading lives here rather than in five copies. The date arithmetic that every
model repeats - how far a purchase sits from the end of its window, and where a
given week starts and ends - lives here for the same reason.

Two things come from the whole store rather than the 6% sample, because a shop
knows them about every customer: how much each article sells, and therefore which
articles exist to be recommended. ``population_sales`` and ``available_articles``
only ever return days up to the end of the history they are given, so a model can
never learn that an article will sell, or even that it exists, from the week it is
predicting.
"""

from __future__ import annotations

import datetime as dt
from functools import cache

import polars as pl

from src.paths import SAMPLE


def load_transactions(split: str | None = None) -> pl.DataFrame:
    """Sampled transactions, optionally one split of ``train`` / ``val`` / ``test``."""
    path = SAMPLE / "transactions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run python -m src.data.sampling first.")
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


def load_articles() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / "articles.parquet")


def load_customers() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE / "customers.parquet")


def week_bounds(last_day: dt.date, weeks_back: int = 0) -> tuple[dt.date, dt.date]:
    """First and last day of the seven-day week ending ``weeks_back`` weeks before ``last_day``."""
    end = last_day - dt.timedelta(days=7 * weeks_back)
    return end - dt.timedelta(days=6), end


def days_before_end(transactions: pl.DataFrame) -> pl.Expr:
    """Days between each purchase and the last day of ``transactions``, the input to every recency weight."""
    return (pl.lit(transactions["t_dat"].max()) - pl.col("t_dat")).dt.total_days()


def holdout_split(weeks_back: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
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
    start, end = week_bounds(transactions["t_dat"].max(), weeks_back)
    return (
        transactions.filter(pl.col("t_dat") < start),
        transactions.filter((pl.col("t_dat") >= start) & (pl.col("t_dat") <= end)),
    )


def purchases_by_customer(transactions: pl.DataFrame) -> dict[str, list[int]]:
    """``{customer_id: [article_id, ...]}`` in purchase order.

    Used both for ground truth (test split) and for purchase history (train split).
    """
    grouped = transactions.sort("t_dat").group_by("customer_id").agg(pl.col("article_id"))
    return dict(zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list()))


@cache
def load_article_sales() -> pl.DataFrame:
    """Daily sales of every article across all customers in the window: ``t_dat, article_id, sales, revenue``."""
    path = SAMPLE / "article_sales.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run python -m src.data.sampling first.")
    return pl.read_parquet(path)


def catalog() -> list[int]:
    """Every article anyone bought inside the window, the rows item features are built for.

    This includes articles first sold in the test week, so it is never a list of
    candidates on its own: each fit narrows it with ``available_articles``.
    """
    return sorted(load_article_sales()["article_id"].unique().to_list())


def population_sales(history: pl.DataFrame) -> pl.DataFrame:
    """Store-wide daily sales up to and including the last day of ``history``."""
    return load_article_sales().filter(pl.col("t_dat") <= history["t_dat"].max())


def available_articles(history: pl.DataFrame) -> set[int]:
    """Articles anyone had bought by the end of ``history``: what a model fitted on it may recommend.

    An article first sold after that day did not exist yet as far as the model
    can know. Offering only those is what keeps the candidate list from carrying
    the answer: the old catalog was every article bought anywhere in the sample,
    test week included, which told the models which brand new articles were about
    to sell.
    """
    return set(population_sales(history)["article_id"].unique().to_list())
