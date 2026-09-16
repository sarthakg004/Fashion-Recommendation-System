"""Model 5 - two-stage retrieval and ranking.

Models 1-4 each answer "what should this customer see?" alone, and each is wrong
in its own way: popularity ignores the person, ALS ignores the item, content
ignores co-purchase, the two-tower finds good candidates but orders them poorly.
The winning Kaggle solutions did not pick one - they pooled candidates from many
cheap retrievers and trained a ranker to sort the pool. This is that idea at a
readable scale.

Stage one, recall: take the top RECALL_K articles from each of the four models
and union them into one candidate set per customer. The pool is small enough to
score exhaustively and much richer than any single model's list - a candidate that
three models nominate is more interesting than one that a single model ranked
first.

Stage two, ranking: describe every (customer, candidate) pair with a handful of
features and train LightGBM's lambdarank objective on them.

    where it came from   each model's rank for this pair, and how many of the four
                         nominated it at all
    how popular it is    training purchase count and popularity rank
    how fresh it is      days since the article was last bought in training
    repurchase           how often this customer already bought this article, and
                         how active the customer is

The ranker learns on the validation week - candidates generated from models fitted
on train, labels taken from val - and is then applied unchanged to the test week.
Nothing is refitted on the test data.

The ceiling is the pool: an article that no retriever nominated cannot be ranked
back in, so the script reports the pool's oracle recall alongside the score.

Run it directly to build, train, score and append the row to
results/metrics_comparison.csv.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMRanker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_utils import load_transactions, purchases_by_customer
from src.metrics import KS, evaluate, save_result

popularity = importlib.import_module("01_popularity")
als = importlib.import_module("02_collaborative_als")
content = importlib.import_module("03_content_based")
two_tower = importlib.import_module("04_two_tower")

MODEL_NAME = "05_two_stage_ranker"
N_RECOMMENDATIONS = max(KS)
RECALL_K = 50
SOURCES = ("popularity", "als", "content", "tower")
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


def fit_retrievers(train: pl.DataFrame):
    """Fit all four models on the training window once."""
    bestsellers = popularity.top_articles(train)

    matrix, als_items, als_index = als.build_matrix(train)
    als_model, als_weighted = als.fit(matrix)

    article_ids = content.catalog(load_transactions())
    blocks = [content.metadata_features(article_ids), content.image_features(article_ids, content.IMAGE_ENCODER)]
    profiles, content_index = content.customer_profiles(train, article_ids, blocks)

    trained = two_tower.train_model(verbose=False)

    def recall(customers) -> dict[str, dict[str, list[int]]]:
        customers = list(customers)
        return {
            "popularity": {c: bestsellers for c in customers},
            "als": als.recommend(als_model, als_weighted, customers, als_items, als_index, bestsellers),
            "content": content.recommend(
                profiles, blocks, [1 - content.IMAGE_WEIGHT, content.IMAGE_WEIGHT], customers, content_index, article_ids, bestsellers
            ),
            "tower": two_tower.rank(
                trained["user_tower"], trained["item_tower"], trained["item_features"], customers, trained["customer_index"],
                trained["article_ids"], trained["history_sum"], trained["history_weight"], trained["static"],
                trained["fallback"], trained["device"],
            ),
        }

    return recall, bestsellers


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


def build_pool(sources: dict[str, dict[str, list[int]]], customers: list[str]) -> pl.DataFrame:
    """Union of each retriever's top-RECALL_K, with that retriever's rank per pair."""
    pool = None
    for name in SOURCES:
        predictions = sources[name]
        customer_column, article_column, rank_column = [], [], []
        for customer in customers:
            items = predictions[customer][:RECALL_K]
            customer_column.extend([customer] * len(items))
            article_column.extend(items)
            rank_column.extend(range(1, len(items) + 1))
        frame = pl.DataFrame(
            {"customer_id": customer_column, "article_id": article_column, f"rank_{name}": rank_column},
            schema={"customer_id": pl.String, "article_id": pl.Int64, f"rank_{name}": pl.Int32},
        )
        pool = frame if pool is None else pool.join(frame, on=["customer_id", "article_id"], how="full", coalesce=True)
    return pool


def add_features(pool: pl.DataFrame, train: pl.DataFrame, articles: pl.DataFrame, customers: pl.DataFrame) -> pl.DataFrame:
    repurchase = train.group_by("customer_id", "article_id").agg(pl.len().alias("times_bought_before"))
    rank_columns = [f"rank_{name}" for name in SOURCES]

    return (
        pool.with_columns(
            pl.sum_horizontal([pl.col(column).is_not_null().cast(pl.Int32) for column in rank_columns]).alias("n_sources")
        )
        .with_columns([pl.col(column).fill_null(RECALL_K + 1) for column in rank_columns])
        .join(articles, on="article_id", how="left")
        .join(customers, on="customer_id", how="left")
        .join(repurchase, on=["customer_id", "article_id"], how="left")
        .with_columns(
            pl.col("article_purchases").fill_null(0),
            pl.col("article_days_since_sold").fill_null(999),
            pl.col("popularity_rank").fill_null(len(SOURCES) * RECALL_K * 10),
            pl.col("customer_purchases").fill_null(0),
            pl.col("times_bought_before").fill_null(0),
        )
        .sort("customer_id", "article_id")
    )


FEATURES = [
    "rank_popularity",
    "rank_als",
    "rank_content",
    "rank_tower",
    "n_sources",
    "article_purchases",
    "article_days_since_sold",
    "popularity_rank",
    "customer_purchases",
    "times_bought_before",
]


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


def rank_pool(model: LGBMRanker, pool: pl.DataFrame, customers, fallback: list[int]) -> dict[str, list[int]]:
    scored = pool.with_columns(pl.Series("score", model.predict(pool.select(FEATURES).to_numpy())))
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


def run() -> dict:
    """Fit both stages and return everything the notebook wants to look at."""
    train = load_transactions("train")
    val_truth = purchases_by_customer(load_transactions("val"))
    test_truth = purchases_by_customer(load_transactions("test"))

    recall, bestsellers = fit_retrievers(train)
    articles = article_statistics(train, bestsellers)
    customers = customer_statistics(train)

    train_pool = label(add_features(build_pool(recall(val_truth), list(val_truth)), train, articles, customers), val_truth)
    test_pool = add_features(build_pool(recall(test_truth), list(test_truth)), train, articles, customers)

    groups = train_pool.group_by("customer_id", maintain_order=True).len()["len"].to_numpy()
    model = LGBMRanker(**LGBM_PARAMS)
    model.fit(train_pool.select(FEATURES).to_numpy(), train_pool["bought"].to_numpy(), group=groups)

    predictions = rank_pool(model, test_pool, test_truth, bestsellers)
    scores = evaluate(predictions, test_truth)
    save_result(MODEL_NAME, scores)

    return dict(
        scores=scores, predictions=predictions, model=model, test_pool=test_pool, train_pool=train_pool,
        ceiling=oracle_recall(test_pool, test_truth), bestsellers=bestsellers,
        importance=pl.DataFrame({"feature": FEATURES, "gain": model.feature_importances_}).sort("gain", descending=True),
    )


def main() -> dict:
    result = run()
    scores, pool = result["scores"], result["test_pool"]
    print(f"{MODEL_NAME}: pool {pool.height:,} pairs for {scores['n_customers']:,} customers "
          f"({pool.height / scores['n_customers']:.0f} candidates each)")
    print(f"  pool ceiling (oracle recall): {result['ceiling']:.4f}")
    print("  top features: " + ", ".join(f"{row[0]}={int(row[1])}" for row in result["importance"].head(5).iter_rows()))
    for k in KS:
        print(f"  @{k:<4} " + "  ".join(f"{m}={scores[f'{m}@{k}']:.5f}" for m in ("precision", "recall", "hitrate", "ndcg", "map")))
    return scores


if __name__ == "__main__":
    main()
