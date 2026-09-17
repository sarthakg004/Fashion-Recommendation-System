"""Stage one of the two-stage model: pooling candidates from several recommenders.

Every retriever answers the same question - "which articles are worth scoring for
this customer?" - and they are deliberately allowed to be bad at it. Stage two
sorts the pool; stage one only has to make sure the right article is somewhere in
it. What matters here is coverage.

Any ``Recommender`` can be a retriever, since the interface is the same. Each one
is fitted once on the history frame and then asked for candidates for whatever
customers it is given. Unknown customers get an empty list rather than a
popularity fallback, because the bestseller retriever already covers everyone and
duplicating it would waste pool slots.
"""

from __future__ import annotations

import gc

import numpy as np
import polars as pl

from src.evaluation.metrics import KS
from src.models.base import Recommender
from src.models.popularity import PopularityRecommender, RecentBestsellers
from src.models.two_tower import TwoTowerRecommender

N_RECOMMENDATIONS = max(KS)


def default_retrievers() -> list[Recommender]:
    """The pool the two-stage model uses.

    Two retrievers, roughly 1,050 candidates per customer, reaching a recall
    ceiling of about 0.47. Pooling six retrievers at ~500 candidates each reached
    0.32, so this is both smaller in code and larger in coverage: what sells
    right now plus what the two-tower thinks this customer wants covers most of
    what the others were contributing, and both scale to large k where the
    heuristics run out of candidates.

    ``AlsRecommender``, ``ContentRecommender`` and the two classes in
    ``src/models/heuristics.py`` can all be pooled back in, but they are not in
    the default pool today.
    """
    return [RecentBestsellers(k=700), TwoTowerRecommender(k=700)]


def bestseller_list(retrievers: list[Recommender]) -> list[int]:
    """The top of the pool's popularity list, used for padding and the popularity-rank feature."""
    popularity = next(r for r in retrievers if isinstance(r, PopularityRecommender))
    return popularity.ranked[:N_RECOMMENDATIONS]


def candidate_frame(retriever: Recommender, customers) -> pl.DataFrame:
    """One retriever's nominations as (customer_id, article_id, rank_<name>)."""
    ranked = retriever.recommend(customers)
    frame = pl.DataFrame(
        {"customer_id": list(ranked), "article_id": [list(items) for items in ranked.values()]},
        schema={"customer_id": pl.String, "article_id": pl.List(pl.Int64)},
    )
    return (
        frame.explode("article_id")
        .drop_nulls("article_id")
        .with_columns((pl.col("article_id").cum_count().over("customer_id")).cast(pl.Int32).alias(f"rank_{retriever.name}"))
    )


def build_pool(retrievers: list[Recommender], customers) -> pl.DataFrame:
    """Union of every retriever's nominations, carrying each one's rank."""
    pool = None
    for retriever in retrievers:
        frame = candidate_frame(retriever, customers)
        pool = frame if pool is None else pool.join(frame, on=["customer_id", "article_id"], how="full", coalesce=True)
        del frame
        gc.collect()
    return pool


def oracle_recall(pool: pl.DataFrame, truth: dict[str, list[int]]) -> float:
    """Recall a perfect ranker would reach on this pool, averaged per customer.

    Averaged the same way ``evaluate`` averages recall@k, so the two are directly
    comparable: this is the ceiling the ranking stage is working against.
    """
    grouped = pool.group_by("customer_id").agg(pl.col("article_id"))
    reachable = {c: set(items) for c, items in zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list())}
    return float(
        np.mean([len(set(items) & reachable.get(customer, set())) / len(set(items)) for customer, items in truth.items()])
    )


def take_top(pool: pl.DataFrame, column: str, customers, fallback: list[int], descending: bool) -> dict[str, list[int]]:
    """Best N_RECOMMENDATIONS per customer by one column, padded with the fallback.

    Both stages are read out through this: the ranker sorts on its own score,
    and the retrieval stage on ``best_rank``, which is the order the pool
    arrived in before anything was learned about it.
    """
    ordered = (
        pool.sort(["customer_id", column], descending=[False, descending])
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(N_RECOMMENDATIONS))
    )
    predictions = dict(zip(ordered["customer_id"].to_list(), ordered["article_id"].to_list()))

    complete = {}
    for customer in customers:
        items = predictions.get(customer, [])
        if len(items) < N_RECOMMENDATIONS:
            items = items + [a for a in fallback if a not in set(items)][: N_RECOMMENDATIONS - len(items)]
        complete[customer] = items[:N_RECOMMENDATIONS]
    return complete
