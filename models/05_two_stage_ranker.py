"""Model 5 - two-stage retrieval and ranking.

Models 1-4 each answer "what should this customer see?" alone, and each is wrong
in its own way: popularity ignores the person, ALS ignores the item, content
ignores co-purchase, the two-tower finds good candidates but orders them poorly.
The winning Kaggle solutions did not pick one - they pooled candidates from many
cheap retrievers and trained a ranker to sort the pool. This is that idea at a
readable scale.

Stage one, recall: every retriever in ``src/retrievers.py`` nominates its top
candidates and the union becomes the candidate set. Three of the six are simple
heuristics rather than models - what sold last week, what this customer already
bought, and other colourways of it - and they matter more than the trained models
do. A pool of the four models alone reached a recall ceiling of 0.085, and the ranker
extracted 97% of it - ranking was saturated and recall was the binding constraint.
Adding the three heuristics lifts the ceiling to about 0.32, of which the ranker
now reaches roughly half. The constraint has moved from recall to ordering, which
is why the next gain should come from richer features rather than more candidates.

Stage two, ranking: describe every (customer, candidate) pair with a handful of
features and train LightGBM's lambdarank objective on them.

    where it came from   each retriever's rank for this pair, and how many of them
                         nominated it at all
    how popular it is    training purchase count and popularity rank
    how fresh it is      days since the article was last bought in training
    repurchase           how often this customer already bought this article, and
                         how active the customer is

The ranker learns from LABEL_WEEKS consecutive weeks, not one. Each training week
is built with a rolling origin: retrievers are refitted on everything strictly
before that week, asked for candidates, and labelled with what the customer
actually bought during it. Refitting per week is the slow part of this script and
it is not optional - reusing one fitted model across all the label weeks would let
it nominate candidates using purchases from after the week it is being scored on.

One week of labels was not enough. With the enlarged pool the ranker saw roughly
13k positives against 519 candidates per customer, and every extra tree memorised
that single week: more capacity and a binary objective both scored worse. Stacking
several weeks is what lets the pool pay off.

The most recent label week is the validation week, whose history is exactly the
training frame used at test time, so the ranker is applied to test under the same
conditions it was trained on. Nothing is refitted on the test data.

The ceiling is still the pool: an article that no retriever nominated cannot be
ranked back in, so the script reports the pool's oracle recall alongside the score.

Memory is the constraint that shapes this file. Candidates are built and scored in
batches of customers, negatives are dropped before the feature joins rather than
after, and polars is held to a few worker threads, because one thread per core
each with its own buffers is what makes a pool this size run out of room. Peak
usage is a little over 2.5 GB regardless of how many candidates each retriever
nominates.

Run it directly to build, train, score and append the row to
results/metrics_comparison.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import datetime as dt
import gc
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMRanker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_utils import load_fitting_data, load_transactions, purchases_by_customer
from src.metrics import KS, evaluate, save_result
from src.retrievers import Retriever, default_retrievers

MODEL_NAME = "05_two_stage_ranker"
N_RECOMMENDATIONS = max(KS)
MISSING_RANK = 9999
LABEL_WEEKS = 4
NEGATIVES_PER_CUSTOMER = 30
CUSTOMER_BATCH = 300
SEED = 42
LGBM_PARAMS = dict(
    objective="lambdarank",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbose=-1,
)

STATIC_FEATURES = [
    "n_sources",
    "best_rank",
    "article_purchases",
    "article_days_since_sold",
    "popularity_rank",
    "customer_purchases",
    "times_bought_before",
]


def feature_names(retrievers: list[Retriever]) -> list[str]:
    return [f"rank_{r.name}" for r in retrievers] + STATIC_FEATURES


def candidate_frame(retriever: Retriever, customers: list[str]) -> pl.DataFrame:
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


def build_pool(retrievers: list[Retriever], customers: list[str]) -> pl.DataFrame:
    """Union of every retriever's nominations, carrying each one's rank."""
    pool = None
    for retriever in retrievers:
        frame = candidate_frame(retriever, customers)
        pool = frame if pool is None else pool.join(frame, on=["customer_id", "article_id"], how="full", coalesce=True)
        del frame
        gc.collect()
    return pool


def article_statistics(train: pl.DataFrame, bestsellers: list[int]) -> pl.DataFrame:
    last_day = train["t_dat"].max()
    stats = train.group_by("article_id").agg(
        pl.len().alias("article_purchases"),
        ((pl.lit(last_day) - pl.col("t_dat").max()).dt.total_days()).alias("article_days_since_sold"),
    )
    ranks = pl.DataFrame({"article_id": bestsellers, "popularity_rank": range(1, len(bestsellers) + 1)})
    return stats.join(ranks, on="article_id", how="full", coalesce=True)


def customer_statistics(train: pl.DataFrame) -> pl.DataFrame:
    return train.group_by("customer_id").agg(pl.len().alias("customer_purchases"))


def add_features(pool: pl.DataFrame, train: pl.DataFrame, articles: pl.DataFrame, customers: pl.DataFrame,
                 rank_columns: list[str]) -> pl.DataFrame:
    repurchase = train.group_by("customer_id", "article_id").agg(pl.len().alias("times_bought_before"))

    return (
        pool.with_columns(
            pl.sum_horizontal([pl.col(column).is_not_null().cast(pl.Int32) for column in rank_columns]).alias("n_sources"),
            pl.min_horizontal([pl.col(column) for column in rank_columns]).fill_null(MISSING_RANK).cast(pl.Int32).alias("best_rank"),
        )
        .with_columns([pl.col(column).fill_null(MISSING_RANK) for column in rank_columns])
        .join(articles, on="article_id", how="left")
        .join(customers, on="customer_id", how="left")
        .join(repurchase, on=["customer_id", "article_id"], how="left")
        .with_columns(
            pl.col("article_purchases").fill_null(0),
            pl.col("article_days_since_sold").fill_null(999),
            pl.col("popularity_rank").fill_null(MISSING_RANK),
            pl.col("customer_purchases").fill_null(0),
            pl.col("times_bought_before").fill_null(0),
        )
        .sort("customer_id", "article_id")
    )


def label(pool: pl.DataFrame, truth: dict[str, list[int]]) -> pl.DataFrame:
    bought = pl.DataFrame(
        {
            "customer_id": [c for c, items in truth.items() for _ in set(items)],
            "article_id": [a for items in truth.values() for a in set(items)],
        },
        schema={"customer_id": pl.String, "article_id": pl.Int64},
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("bought"))
    return pool.join(bought, on=["customer_id", "article_id"], how="left").with_columns(pl.col("bought").fill_null(0))


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


def rank_pool(model: LGBMRanker, pool: pl.DataFrame, customers, features: list[str], fallback: list[int]) -> dict[str, list[int]]:
    scored = pool.with_columns(pl.Series("score", model.predict(pool.select(features).to_numpy())))
    ordered = (
        scored.sort(["customer_id", "score"], descending=[False, True])
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


def downsample(pool: pl.DataFrame, negatives_per_customer: int = NEGATIVES_PER_CUSTOMER) -> pl.DataFrame:
    """Keep every positive and a few negatives per customer.

    A full pool is a fraction of a percent positives, which is what the ranker
    was drowning in: it scored worse the more capacity it was given. Keeping all
    positives plus a sample of negatives, and dropping customers who have no
    positive at all because they carry no ordering signal, cuts the training rows
    by a large factor and scores better.
    """
    positives = pool.filter(pl.col("bought") == 1)
    negatives = (
        pool.filter((pl.col("bought") == 0) & pl.col("customer_id").is_in(positives["customer_id"].unique()))
        .sample(fraction=1.0, shuffle=True, seed=SEED)
        .with_columns(pl.int_range(pl.len()).over("customer_id").alias("draw"))
        .filter(pl.col("draw") < negatives_per_customer)
        .drop("draw")
    )
    return pl.concat([positives, negatives])


def label_weeks(n_weeks: int = LABEL_WEEKS):
    """Rolling origin: each week paired with the history available before it."""
    transactions = load_transactions()
    labelled = transactions.filter(pl.col("split") != "test")
    last_day = labelled["t_dat"].max()

    for index in range(n_weeks):
        end = last_day - dt.timedelta(days=7 * index)
        start = end - dt.timedelta(days=6)
        week = labelled.filter((pl.col("t_dat") >= start) & (pl.col("t_dat") <= end))
        history = labelled.filter(pl.col("t_dat") < start)
        yield index, history, purchases_by_customer(week)


def build_training_pool(n_weeks: int = LABEL_WEEKS, verbose: bool = True) -> pl.DataFrame:
    """Stack several labelled weeks, each built from its own history.

    Two things keep this inside memory. Negatives are dropped before the feature
    joins, because all but a few percent of a week's pool is thrown away and
    joining statistics onto the full thing first is wasted work. And customers
    are processed in batches, so peak memory depends on CUSTOMER_BATCH rather
    than on how many candidates each retriever nominates.
    """
    weeks = []
    for index, history, truth in label_weeks(n_weeks):
        retrievers = [r.fit(history) for r in default_retrievers()]
        rank_columns = [f"rank_{r.name}" for r in retrievers]
        bestsellers = retrievers[0].recommend(["_"])["_"][:N_RECOMMENDATIONS]
        articles, customers = article_statistics(history, bestsellers), customer_statistics(history)

        batches, raw_pairs, positives = [], 0, 0
        for batch in batched(list(truth), CUSTOMER_BATCH):
            candidates = label(build_pool(retrievers, batch), truth)
            raw_pairs += candidates.height
            positives += int(candidates["bought"].sum())
            batches.append(add_features(downsample(candidates), history, articles, customers, rank_columns))
            del candidates
            gc.collect()

        weekly = pl.concat(batches).with_columns(pl.lit(index, dtype=pl.Int32).alias("week"))
        weeks.append(weekly)
        if verbose:
            print(f"  label week -{index}: {len(truth):,} customers, {raw_pairs:,} pairs, "
                  f"{positives:,} positives -> {weekly.height:,} training rows")
        del batches, retrievers
        gc.collect()
    return pl.concat(weeks).sort("week", "customer_id")


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run(retrievers: list[Retriever] | None = None, n_weeks: int = LABEL_WEEKS, verbose: bool = True,
        save: bool = True) -> dict:
    """Fit both stages and return everything the notebook wants to look at."""
    retrievers = retrievers or default_retrievers()
    fitting = load_fitting_data()
    test_truth = purchases_by_customer(load_transactions("test"))

    train_pool = build_training_pool(n_weeks, verbose)

    for retriever in retrievers:
        retriever.fit(fitting)

    features = feature_names(retrievers)
    rank_columns = [f"rank_{r.name}" for r in retrievers]
    bestsellers = retrievers[0].recommend(["_"])["_"][:N_RECOMMENDATIONS]
    articles, customers = article_statistics(fitting, bestsellers), customer_statistics(fitting)
    pool_rows = 0

    groups = train_pool.group_by("week", "customer_id", maintain_order=True).len()["len"].to_numpy()
    model = LGBMRanker(**LGBM_PARAMS)
    model.fit(train_pool.select(features).to_numpy(), train_pool["bought"].to_numpy(), group=groups)

    predictions, reachable = {}, []
    for batch in batched(list(test_truth), CUSTOMER_BATCH):
        pool = build_pool(retrievers, batch)
        reachable.append(oracle_recall(pool, {c: test_truth[c] for c in batch}))
        featured = add_features(pool, fitting, articles, customers, rank_columns)
        predictions.update(rank_pool(model, featured, batch, features, bestsellers))
        pool_rows += featured.height
        del pool, featured
        gc.collect()

    ceiling = float(np.mean(reachable))
    scores = evaluate(predictions, test_truth)
    if save:
        save_result(MODEL_NAME, scores)

    return dict(
        scores=scores, predictions=predictions, model=model, train_pool=train_pool, pool_rows=pool_rows,
        ceiling=ceiling, bestsellers=bestsellers, retrievers=retrievers,
        recall_by_retriever=pl.DataFrame(
            [
                {"retriever": r.name, "k": r.k,
                 "ceiling": round(np.mean([oracle_recall(candidate_frame(r, batch), {c: test_truth[c] for c in batch})
                                           for batch in batched(list(test_truth), CUSTOMER_BATCH)]), 4)}
                for r in retrievers
            ]
        ).sort("ceiling", descending=True),
        importance=pl.DataFrame({"feature": features, "gain": model.feature_importances_}).sort("gain", descending=True),
    )


def main() -> dict:
    result = run()
    scores, pool_rows = result["scores"], result["pool_rows"]
    print(f"{MODEL_NAME}: pool {pool_rows:,} pairs for {scores['n_customers']:,} customers "
          f"({pool_rows / scores['n_customers']:.0f} candidates each)")
    print(f"  pool ceiling (oracle recall): {result['ceiling']:.4f}")
    print("  top features: " + ", ".join(f"{row[0]}={int(row[1])}" for row in result["importance"].head(5).iter_rows()))
    for k in KS:
        print(f"  @{k:<4} " + "  ".join(f"{m}={scores[f'{m}@{k}']:.5f}" for m in ("precision", "recall", "hitrate", "ndcg", "map")))
    return scores


if __name__ == "__main__":
    main()
