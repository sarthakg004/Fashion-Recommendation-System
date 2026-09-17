"""Stage two's inputs: the features describing every (customer, candidate) pair.

    where it came from   each retriever's rank for this pair, how many nominated
                         it, and the best rank any of them gave it
    how it is selling    lifetime purchases, last week's purchases, and the ratio
                         of last week to the week before. A coat selling three
                         times faster than last week is a different proposition
                         from one with the same total that is fading.
    how fresh it is      days since the article last sold
    what it costs        the article's average price, and that price relative to
                         what this customer usually spends, which stops a pool
                         full of cheap bestsellers being offered to someone who
                         only buys expensive coats
    the customer         how much they buy, how recently, over how many days
    affinity             how often they bought this exact article, this garment in
                         another colour (a shared product_code), and this product
                         type. The exact article is the narrowest of the three and
                         was the only one the ranker used to see.

Every statistic is computed from the history frame the retrievers were fitted on,
never from the week being labelled or scored. The order of ``feature_names`` is
the column order the ranker is trained and applied on.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from src.data.loading import load_articles
from src.models.base import Recommender

MISSING_RANK = 9999

STATIC_FEATURES = [
    "n_sources",
    "best_rank",
    "article_purchases",
    "article_recent_purchases",
    "article_velocity",
    "article_days_since_sold",
    "article_price",
    "popularity_rank",
    "customer_purchases",
    "customer_days_since_purchase",
    "customer_active_days",
    "price_ratio",
    "times_bought_before",
    "product_bought_before",
    "type_bought_before",
]


def rank_columns(retrievers: list[Recommender]) -> list[str]:
    return [f"rank_{r.name}" for r in retrievers]


def feature_names(retrievers: list[Recommender]) -> list[str]:
    return rank_columns(retrievers) + STATIC_FEATURES


def article_statistics(history: pl.DataFrame, bestsellers: list[int]) -> pl.DataFrame:
    """How much an article sells, how recently, and whether it is on the way up.

    ``article_velocity`` is last week's sales over the week before. A coat selling
    three times as fast as it did last week is a different proposition from one
    with the same total that is fading, and total counts cannot express that.
    """
    last_day = history["t_dat"].max()
    recent = history.filter(pl.col("t_dat") > last_day - dt.timedelta(days=7))
    previous = history.filter(
        (pl.col("t_dat") <= last_day - dt.timedelta(days=7)) & (pl.col("t_dat") > last_day - dt.timedelta(days=14))
    )

    stats = history.group_by("article_id").agg(
        pl.len().alias("article_purchases"),
        ((pl.lit(last_day) - pl.col("t_dat").max()).dt.total_days()).alias("article_days_since_sold"),
        pl.col("price").mean().alias("article_price"),
    )
    weekly = recent.group_by("article_id").agg(pl.len().alias("article_recent_purchases"))
    prior = previous.group_by("article_id").agg(pl.len().alias("article_prior_purchases"))
    ranks = pl.DataFrame({"article_id": bestsellers, "popularity_rank": range(1, len(bestsellers) + 1)})

    return (
        stats.join(weekly, on="article_id", how="full", coalesce=True)
        .join(prior, on="article_id", how="full", coalesce=True)
        .join(ranks, on="article_id", how="full", coalesce=True)
        .with_columns(
            pl.col("article_recent_purchases").fill_null(0),
            pl.col("article_prior_purchases").fill_null(0),
        )
        .with_columns(
            ((pl.col("article_recent_purchases") + 1) / (pl.col("article_prior_purchases") + 1)).alias("article_velocity")
        )
        .drop("article_prior_purchases")
    )


def customer_statistics(history: pl.DataFrame) -> pl.DataFrame:
    """How much a customer buys, how recently, and what they usually spend."""
    last_day = history["t_dat"].max()
    return history.group_by("customer_id").agg(
        pl.len().alias("customer_purchases"),
        ((pl.lit(last_day) - pl.col("t_dat").max()).dt.total_days()).alias("customer_days_since_purchase"),
        pl.col("price").mean().alias("customer_price"),
        pl.col("t_dat").n_unique().alias("customer_active_days"),
    )


def affinity_statistics(history: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """What a customer has bought before, at the product and category level.

    The exact article is the narrowest possible match. A customer who bought a
    garment in black is a strong candidate for the same garment in green, which
    shares a ``product_code``, and a weaker but real one for anything else of that
    product type.
    """
    article_groups = load_articles().select("article_id", "product_code", "product_type_no")
    tagged = history.join(article_groups, on="article_id", how="left")
    by_code = tagged.group_by("customer_id", "product_code").agg(pl.len().alias("product_bought_before"))
    by_type = tagged.group_by("customer_id", "product_type_no").agg(pl.len().alias("type_bought_before"))
    return by_code, by_type


def add_features(pool: pl.DataFrame, history: pl.DataFrame, article_stats: pl.DataFrame,
                 customer_stats: pl.DataFrame, ranks: list[str]) -> pl.DataFrame:
    """Join every feature onto a candidate pool; ``ranks`` are the pool's ``rank_<retriever>`` columns."""
    repurchase = history.group_by("customer_id", "article_id").agg(pl.len().alias("times_bought_before"))
    by_code, by_type = affinity_statistics(history)
    article_groups = load_articles().select("article_id", "product_code", "product_type_no")

    return (
        pool.with_columns(
            pl.sum_horizontal([pl.col(column).is_not_null().cast(pl.Int32) for column in ranks]).alias("n_sources"),
            pl.min_horizontal([pl.col(column) for column in ranks]).fill_null(MISSING_RANK).cast(pl.Int32).alias("best_rank"),
        )
        .with_columns([pl.col(column).fill_null(MISSING_RANK) for column in ranks])
        .join(article_stats, on="article_id", how="left")
        .join(customer_stats, on="customer_id", how="left")
        .join(repurchase, on=["customer_id", "article_id"], how="left")
        .join(article_groups, on="article_id", how="left")
        .join(by_code, on=["customer_id", "product_code"], how="left")
        .join(by_type, on=["customer_id", "product_type_no"], how="left")
        .with_columns(
            pl.col("article_purchases").fill_null(0),
            pl.col("article_recent_purchases").fill_null(0),
            pl.col("article_velocity").fill_null(1.0),
            pl.col("article_days_since_sold").fill_null(999),
            pl.col("article_price").fill_null(0.0),
            pl.col("popularity_rank").fill_null(MISSING_RANK),
            pl.col("customer_purchases").fill_null(0),
            pl.col("customer_days_since_purchase").fill_null(999),
            pl.col("customer_price").fill_null(0.0),
            pl.col("customer_active_days").fill_null(0),
            pl.col("times_bought_before").fill_null(0),
            pl.col("product_bought_before").fill_null(0),
            pl.col("type_bought_before").fill_null(0),
        )
        .with_columns(
            (pl.col("article_price") / pl.when(pl.col("customer_price") > 0).then(pl.col("customer_price")).otherwise(None))
            .fill_null(1.0)
            .alias("price_ratio")
        )
        .drop("product_code", "product_type_no")
        .sort("customer_id", "article_id")
    )
