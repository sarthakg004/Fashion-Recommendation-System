"""Model 1 - popularity baseline.

The simplest thing that can be called a recommender: count how many times each
article was bought during the training weeks, sort by that count, and hand the
same top-100 list to every customer. There is no personalisation at all - two
customers with nothing in common get identical recommendations.

It exists to set the floor. A personalised model that cannot beat "everyone gets
the bestsellers" has not earned its complexity, and in a catalog this skewed the
floor is higher than it sounds: a handful of basics account for a large share of
all purchases.

Run it directly to score it and append the row to results/metrics_comparison.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_fitting_data, load_transactions, purchases_by_customer
from src.metrics import KS, evaluate, save_result

MODEL_NAME = "01_popularity"
N_RECOMMENDATIONS = max(KS)


def top_articles(train: pl.DataFrame, n: int = N_RECOMMENDATIONS) -> list[int]:
    return (
        train.group_by("article_id")
        .agg(pl.len().alias("purchases"))
        .sort("purchases", descending=True)
        .head(n)["article_id"]
        .to_list()
    )


def recommend(customers, ranked: list[int]) -> dict[str, list[int]]:
    return {customer: ranked for customer in customers}


def main() -> dict:
    train, test = load_fitting_data(), load_transactions("test")
    ground_truth = purchases_by_customer(test)

    ranked = top_articles(train)
    assert len(ranked) == N_RECOMMENDATIONS, len(ranked)

    predictions = recommend(ground_truth, ranked)
    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)

    print(f"{MODEL_NAME}: {scores['n_customers']:,} test customers")
    for k in KS:
        print(f"  @{k:<4} " + "  ".join(f"{m}={scores[f'{m}@{k}']:.5f}" for m in ("precision", "recall", "hitrate", "ndcg", "map")))
    return scores


if __name__ == "__main__":
    main()
