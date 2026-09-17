"""Model 2 - collaborative filtering with ALS.

The first personalised model. It ignores everything about what an article *is* -
no photos, no descriptions, no categories - and looks only at who bought what.
Purchases go into a big sparse customer x article matrix, and ALS factorises it
into a small dense vector per customer and per article, chosen so that their dot
product reproduces the observed purchases. A customer's recommendations are the
articles whose vectors point the same way as theirs.

What that buys over popularity: the model discovers that people who buy this tend
to also buy that, without anyone describing either item. What it cannot do is say
anything about an article nobody has bought yet - a brand new article has no
column in the matrix, which is the cold-start problem model 3 exists to solve.

Three settings that matter, all chosen on the validation week:

    time decay       a purchase counts as alpha / (1 + days_before_the_split),
                     so last week's basket weighs far more than one from four
                     months ago. Fashion moves fast and this is the single
                     largest win here (val MAP@12 0.0141 -> 0.0241).
    BM25 weighting   down-weights the bestsellers everyone buys, so the factors
                     spend their capacity on informative co-purchases
                     (val MAP@12 0.0102 -> 0.0141 on undecayed counts).
    no filtering of already-bought items   fashion is heavily repurchased, and
                     suppressing known items cost roughly half the score.

The time-decay confidence is borrowed from JonMcEntee/hm-fashion-recommendations,
which applies the same alpha / (1 + days) weighting before fitting ALS. Alpha was
re-tuned here and plateaus between 40 and 80; exponential decay scored the same,
so the simpler form stays.

Customers with no training history have no vector at all; they fall back to the
popularity list from model 1, which is what a production system would do.

Run it with ``python -m src.models.als`` to score it and append the row to
results/metrics_comparison.csv.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from threadpoolctl import threadpool_limits

from src.data.loading import days_before_end, load_fitting_data, load_transactions, purchases_by_customer
from src.evaluation.metrics import KS, evaluate
from src.evaluation.reporting import print_scores, save_result
from src.models.base import Recommender
from src.models.popularity import PopularityRecommender

MODEL_NAME = "02_collaborative_als"
N_RECOMMENDATIONS = max(KS)
FACTORS = 512
REGULARIZATION = 0.05
ITERATIONS = 15
BM25_K1 = 100
BM25_B = 0.8
TIME_DECAY_ALPHA = 40.0
SEED = 42


def build_matrix(train: pl.DataFrame):
    """Sparse customer x article matrix of time-decayed purchase confidence.

    A purchase made ``d`` days before the end of the training window contributes
    ``TIME_DECAY_ALPHA / (1 + d)``, and repeat purchases of the same article add
    up. Recency is what the decay buys: the same purchase is worth 40x more on
    the last day of the window than on the 40th day back.

    Returns the matrix, the article id of each column, and each customer's row.
    """
    users = train["customer_id"].unique().sort().to_list()
    items = train["article_id"].unique().sort().to_list()
    user_index = {user: i for i, user in enumerate(users)}
    item_index = {item: i for i, item in enumerate(items)}

    days_before_split = train.select(days_before_end(train).alias("days"))["days"].to_numpy()
    confidence = (TIME_DECAY_ALPHA / (1.0 + days_before_split)).astype(np.float32)

    matrix = sp.csr_matrix(
        (
            confidence,
            (train["customer_id"].replace_strict(user_index).to_numpy(), train["article_id"].replace_strict(item_index).to_numpy()),
        ),
        shape=(len(users), len(items)),
    )
    matrix.sum_duplicates()
    return matrix, items, user_index


class AlsRecommender(Recommender):
    """BM25-weighted, time-decayed ALS over the customer x article matrix."""

    name = "als"

    def fit(self, train: pl.DataFrame) -> "AlsRecommender":
        self.matrix, self.items, self.user_index = build_matrix(train)
        self.weighted = bm25_weight(self.matrix, K1=BM25_K1, B=BM25_B).tocsr()
        self.model = AlternatingLeastSquares(
            factors=FACTORS, regularization=REGULARIZATION, iterations=ITERATIONS, random_state=SEED, num_threads=8
        )
        with threadpool_limits(1, "blas"):
            self.model.fit(self.weighted, show_progress=False)
        return self

    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        """Top-k articles per customer. ``k`` is capped at the catalog size."""
        known = [c for c in customers if c in self.user_index]
        rows = np.array([self.user_index[c] for c in known])
        ranked, _ = self.model.recommend(
            rows, self.weighted[rows], N=min(self.k, len(self.items)), filter_already_liked_items=False
        )
        personalised = {c: [self.items[j] for j in row] for c, row in zip(known, ranked)}
        return {c: personalised.get(c, fallback) for c in customers}


def main() -> dict:
    train, test = load_fitting_data(), load_transactions("test")
    ground_truth = purchases_by_customer(test)

    model = AlsRecommender(k=N_RECOMMENDATIONS).fit(train)
    fallback = PopularityRecommender().fit(train).ranked

    predictions = model.recommend(ground_truth, fallback)
    covered = sum(1 for c in ground_truth if c in model.user_index)
    assert all(len(p) == N_RECOMMENDATIONS for p in predictions.values())

    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)

    print(f"{MODEL_NAME}: matrix {model.matrix.shape} nnz={model.matrix.nnz:,}")
    print(f"  personalised for {covered:,} / {len(ground_truth):,} test customers ({100 * covered / len(ground_truth):.0f}%), rest fall back to popularity")
    print_scores(scores)
    return scores


if __name__ == "__main__":
    main()
