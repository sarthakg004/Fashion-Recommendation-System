"""Model 1 - popularity baseline.

The simplest thing that can be called a recommender: count how many times each
article was bought during the training weeks, sort by that count, and hand the
same top-100 list to every customer. There is no personalisation at all - two
customers with nothing in common get identical recommendations.

It exists to set the floor. A personalised model that cannot beat "everyone gets
the bestsellers" has not earned its complexity, and in a catalog this skewed the
floor is higher than it sounds: a handful of basics account for a large share of
all purchases.

The same count restricted to the last few days is ``RecentBestsellers``, the
two-stage model's strongest retriever: what is selling right now beats any model
trained on a four-month average (recall 0.178 at k=300 on the test week).

Run it with ``python -m src.models.popularity`` to score it and append the row to
results/metrics_comparison.csv.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from src.data.loading import load_fitting_data, load_transactions, purchases_by_customer
from src.evaluation.metrics import KS, evaluate
from src.evaluation.reporting import print_scores, save_result
from src.models.base import Recommender

MODEL_NAME = "01_popularity"
N_RECOMMENDATIONS = max(KS)


class PopularityRecommender(Recommender):
    """The most bought articles, optionally only over the final ``days`` of the frame."""

    name = "popularity"
    default_k = N_RECOMMENDATIONS

    def __init__(self, days: int | None = None, k: int | None = None):
        super().__init__(k)
        self.days = days
        self.ranked: list[int] = []

    def fit(self, train: pl.DataFrame) -> "PopularityRecommender":
        if self.days is not None:
            train = train.filter(pl.col("t_dat") > train["t_dat"].max() - pl.duration(days=self.days))
        self.ranked = (
            train.group_by("article_id")
            .agg(pl.len().alias("purchases"))
            .sort("purchases", descending=True)
            .head(self.k)["article_id"]
            .to_list()
        )
        return self

    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        return {customer: self.ranked for customer in customers}


class RecentBestsellers(PopularityRecommender):
    """What sold most in the final days of training, same list for everyone."""

    name = "recent"
    default_k = 300

    def __init__(self, days: int = 7, k: int | None = None):
        super().__init__(days, k)


def main() -> dict:
    train, test = load_fitting_data(), load_transactions("test")
    ground_truth = purchases_by_customer(test)

    model = PopularityRecommender().fit(train)
    assert len(model.ranked) == N_RECOMMENDATIONS, len(model.ranked)

    predictions = model.recommend(ground_truth)
    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)

    print(f"{MODEL_NAME}: {scores['n_customers']:,} test customers")
    print_scores(scores)
    return scores


if __name__ == "__main__":
    main()
