"""Model 5 - two-stage retrieval and ranking.

Models 1-4 each answer "what should this customer see?" alone, and each is wrong
in its own way: popularity ignores the person, ALS ignores the item, content
ignores co-purchase, the two-tower finds good candidates but orders them poorly.
The winning Kaggle solutions did not pick one - they pooled candidates from many
cheap retrievers and trained a ranker to sort the pool. This is that idea at a
readable scale.

Stage one, recall: two retrievers nominate 700 candidates each and the union
becomes the candidate set - what sold in the last seven days, which nothing here
beats on a catalog that turns over weekly, and the two-tower, the strongest
learned retriever of the five. Earlier versions pooled six retrievers at 500
candidates each and reached a recall ceiling of 0.32; these two at 700 reach about
0.47 on a third of the code, because the other four were mostly nominating
articles these two already had. The rest are kept because they stay cheap to pool
back in, not because the default needs them. The pool is built in
``src/ranking/candidates.py``.

The ceiling is what the pool makes reachable, and roughly half of test purchases
are not in it at any size tried. Retrieval, not ranking, is still the larger loss
in this system.

Stage two, ranking: describe every (customer, candidate) pair with seventeen
features and train LightGBM's lambdarank objective on them. Each tree corrects
what the trees before it got wrong, and the learning rate decides how much of
that correction to apply.

The ranker's own settings turned out not to matter much. On the validation week
2500 trees at 0.015 looked clearly best, but on the test week 600, 1200 and 2500
trees land within 0.0004 MAP@12 of each other, which is inside this pipeline's
run-to-run variance. 600 is kept because it is the cheapest of three equivalent
options, not because it scored highest. What did matter was the features: going
from nine to seventeen moved test MAP@12 by about 9%, and three of the five
strongest features by gain are among the new ones. The features themselves, and
why each exists, are described in ``src/ranking/features.py``.

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

Both stages are scored, not just the final one. Reading the pool out in its own
arrival order - by ``best_rank``, before the ranker has said anything - gives the
score the retrieval stage reaches alone, and the difference between that and the
reranked score is what stage two is actually worth. Customers are also scored in
three groups by how much history the model had on them, because an average over
everyone hides that this is a much easier job for someone with forty purchases
behind them than for someone with none.

Accuracy alone would not notice a recommender that has learned to show everybody
the same few hundred bestsellers, so ``beyond_accuracy`` reports what share of
the catalog the top-12 lists touch, how obvious the articles in them are, and how
many distinct product types a single list contains.

One week scored once is a single draw. ``src/evaluation/cross_validation.py``
re-runs this whole pipeline over several held-out weeks and several seeds, which
is what separates a real gain from the pipeline's own noise - and that noise is not
negligible, since the two-tower is retrained per fold on a GPU and repeated runs of
the identical configuration spread with a standard deviation of about 0.0008
MAP@12.

Run it with ``python -m src.ranking.two_stage`` to build, train, score and append
the row to results/metrics_comparison.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import gc
from dataclasses import dataclass
from itertools import batched

import numpy as np
import polars as pl
from lightgbm import LGBMRanker

from src.data.loading import holdout_split, load_articles, purchases_by_customer, week_bounds
from src.evaluation.metrics import KS, beyond_accuracy, evaluate, evaluate_segments
from src.evaluation.reporting import print_scores, save_result, write_report
from src.models.base import Recommender
from src.ranking.candidates import (
    bestseller_list,
    build_pool,
    candidate_frame,
    default_retrievers,
    oracle_recall,
    take_top,
)
from src.ranking.features import add_features, article_statistics, customer_statistics, feature_names, rank_columns

MODEL_NAME = "05_two_stage_ranker"
N_RECOMMENDATIONS = max(KS)
LABEL_WEEKS = 4
NEGATIVES_PER_CUSTOMER = 60
CUSTOMER_BATCH = 300
SEED = 42
LGBM_PARAMS = dict(
    objective="lambdarank",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbose=-1,
)


def attach_labels(pool: pl.DataFrame, truth: dict[str, list[int]]) -> pl.DataFrame:
    """A ``bought`` column: 1 if the customer bought the candidate during the labelled week."""
    bought = pl.DataFrame(
        {
            "customer_id": [c for c, items in truth.items() for _ in set(items)],
            "article_id": [a for items in truth.values() for a in set(items)],
        },
        schema={"customer_id": pl.String, "article_id": pl.Int64},
    ).with_columns(pl.lit(1, dtype=pl.Int8).alias("bought"))
    return pool.join(bought, on=["customer_id", "article_id"], how="left").with_columns(pl.col("bought").fill_null(0))


def downsample(pool: pl.DataFrame, negatives_per_customer: int = NEGATIVES_PER_CUSTOMER, seed: int = SEED) -> pl.DataFrame:
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
        .sample(fraction=1.0, shuffle=True, seed=seed)
        .with_columns(pl.int_range(pl.len()).over("customer_id").alias("draw"))
        .filter(pl.col("draw") < negatives_per_customer)
        .drop("draw")
    )
    return pl.concat([positives, negatives])


def label_weeks(n_weeks: int, fitting: pl.DataFrame):
    """Rolling origin: each week paired with the history available before it.

    ``fitting`` is everything the model may learn from, so its final week is the
    most recent week that can be labelled.
    """
    last_day = fitting["t_dat"].max()

    for index in range(n_weeks):
        start, end = week_bounds(last_day, index)
        week = fitting.filter((pl.col("t_dat") >= start) & (pl.col("t_dat") <= end))
        history = fitting.filter(pl.col("t_dat") < start)
        yield index, history, purchases_by_customer(week)


def customer_segments(fitting: pl.DataFrame, truth: dict[str, list[int]]) -> dict[str, set[str]]:
    """Group the evaluated customers by how much history the model had on them.

    Recommending to someone with forty purchases behind them is a different job
    from recommending to someone with none, and a single average silently mixes
    the two. Cold customers here are the ones the pipeline has never seen buy
    anything, so only the bestseller retriever can reach them at all.
    """
    counted = fitting.group_by("customer_id").agg(pl.len().alias("n"))
    counts = dict(zip(counted["customer_id"].to_list(), counted["n"].to_list()))
    groups: dict[str, set[str]] = {"cold (no history)": set(), "light (1-4)": set(), "heavy (5+)": set()}
    for customer in truth:
        n = counts.get(customer, 0)
        groups["cold (no history)" if n == 0 else "light (1-4)" if n < 5 else "heavy (5+)"].add(customer)
    return {name: members for name, members in groups.items() if members}


class TwoStageRanker:
    """Retrievers nominate, LightGBM's lambdarank orders.

    ``fit`` builds the rolling-origin training pool, refits the retrievers on the
    whole fitting frame and trains the ranker. ``predict`` then pools, features and
    ranks the customers it is given, a batch at a time.
    """

    def __init__(self, retrievers: list[Recommender] | None = None, n_weeks: int = LABEL_WEEKS,
                 seed: int = SEED, verbose: bool = True):
        self.retrievers = retrievers or default_retrievers()
        self.n_weeks = n_weeks
        self.seed = seed
        self.verbose = verbose

    def build_training_pool(self, fitting: pl.DataFrame) -> pl.DataFrame:
        """Stack several labelled weeks, each built from its own history.

        Two things keep this inside memory. Negatives are dropped before the feature
        joins, because all but a few percent of a week's pool is thrown away and
        joining statistics onto the full thing first is wasted work. And customers
        are processed in batches, so peak memory depends on CUSTOMER_BATCH rather
        than on how many candidates each retriever nominates.
        """
        weeks = []
        for index, history, truth in label_weeks(self.n_weeks, fitting):
            retrievers = [r.fit(history) for r in default_retrievers()]
            ranks = rank_columns(retrievers)
            bestsellers = bestseller_list(retrievers)
            article_stats, customer_stats = article_statistics(history, bestsellers), customer_statistics(history)

            batches, raw_pairs, positives = [], 0, 0
            for batch in batched(list(truth), CUSTOMER_BATCH):
                candidates = attach_labels(build_pool(retrievers, batch), truth)
                raw_pairs += candidates.height
                positives += int(candidates["bought"].sum())
                batches.append(add_features(downsample(candidates, seed=self.seed), history, article_stats, customer_stats, ranks))
                del candidates
                gc.collect()

            weekly = pl.concat(batches).with_columns(pl.lit(index, dtype=pl.Int32).alias("week"))
            weeks.append(weekly)
            if self.verbose:
                print(f"  label week -{index}: {len(truth):,} customers, {raw_pairs:,} pairs, "
                      f"{positives:,} positives -> {weekly.height:,} training rows")
            del batches, retrievers
            gc.collect()
        return pl.concat(weeks).sort("week", "customer_id")

    def fit(self, fitting: pl.DataFrame) -> "TwoStageRanker":
        self.fitting = fitting
        self.train_pool = self.build_training_pool(fitting)

        for retriever in self.retrievers:
            retriever.fit(fitting)

        self.features = feature_names(self.retrievers)
        self.ranks = rank_columns(self.retrievers)
        self.bestsellers = bestseller_list(self.retrievers)
        self.article_stats, self.customer_stats = article_statistics(fitting, self.bestsellers), customer_statistics(fitting)

        groups = self.train_pool.group_by("week", "customer_id", maintain_order=True).len()["len"].to_numpy()
        self.model = LGBMRanker(**{**LGBM_PARAMS, "random_state": self.seed})
        self.model.fit(self.train_pool.select(self.features).to_numpy(), self.train_pool["bought"].to_numpy(), group=groups)
        return self

    def predict(self, truth: dict[str, list[int]]):
        """Rank the pool for every customer in ``truth``.

        Returns the reranked top lists, the same pool read out in arrival order
        (by ``best_rank``), the pool's oracle recall, and how many candidate pairs
        were scored. The truth is only used for the oracle recall, never to rank.
        """
        predictions, retrieved, reachable, pool_rows = {}, {}, [], 0
        for batch in batched(list(truth), CUSTOMER_BATCH):
            pool = build_pool(self.retrievers, batch)
            reachable.append(oracle_recall(pool, {c: truth[c] for c in batch}))
            featured = add_features(pool, self.fitting, self.article_stats, self.customer_stats, self.ranks)
            scored = featured.with_columns(pl.Series("score", self.model.predict(featured.select(self.features).to_numpy())))
            predictions.update(take_top(scored, "score", batch, self.bestsellers, descending=True))
            retrieved.update(take_top(featured, "best_rank", batch, self.bestsellers, descending=False))
            pool_rows += featured.height
            del pool, featured, scored
            gc.collect()
        return predictions, retrieved, float(np.mean(reachable)), pool_rows


@dataclass
class TwoStageResult:
    """Everything one evaluation produced, for the notebook, the API and cross-validation."""

    ranker: TwoStageRanker
    truth: dict[str, list[int]]
    predictions: dict[str, list[int]]
    scores: dict
    retrieval_scores: dict
    segments: dict
    catalog_shape: dict
    ceiling: float
    pool_rows: int
    weeks_back: int
    seed: int

    def importance(self) -> pl.DataFrame:
        return pl.DataFrame(
            {"feature": self.ranker.features, "gain": self.ranker.model.feature_importances_}
        ).sort("gain", descending=True)

    def recall_by_retriever(self) -> pl.DataFrame:
        """Each retriever's pool ceiling on its own, which is how the default pool was chosen."""
        return pl.DataFrame(
            [
                {"retriever": r.name, "k": r.k,
                 "ceiling": round(np.mean([oracle_recall(candidate_frame(r, batch), {c: self.truth[c] for c in batch})
                                           for batch in batched(list(self.truth), CUSTOMER_BATCH)]), 4)}
                for r in self.ranker.retrievers
            ]
        ).sort("ceiling", descending=True)


def evaluate_two_stage(retrievers: list[Recommender] | None = None, n_weeks: int = LABEL_WEEKS, verbose: bool = True,
                       save: bool = False, weeks_back: int = 0, seed: int = SEED,
                       window_weeks: int | None = None) -> TwoStageResult:
    """Fit both stages on one held-out split and score them.

    ``weeks_back`` chooses which week to hold out, counting back from the test
    week, and ``seed`` drives both the negative sample and the ranker. Varying
    them is how the cross-validation in ``src/evaluation/cross_validation.py``
    separates a real difference from the noise of one week and one random draw.
    ``window_weeks`` caps how much history is used, which is what holds training
    size constant while the evaluation week moves. ``save`` writes the comparison
    row and ``results/two_stage_report.json``.
    """
    retrievers = retrievers or default_retrievers()
    fitting, evaluation = holdout_split(weeks_back, window_weeks)
    truth = purchases_by_customer(evaluation)

    ranker = TwoStageRanker(retrievers, n_weeks, seed, verbose).fit(fitting)
    predictions, retrieved, ceiling, pool_rows = ranker.predict(truth)
    scores = evaluate(predictions, truth)

    counts = fitting.group_by("article_id").agg(pl.len().alias("n"))
    catalog = load_articles().select("article_id", "product_type_no")
    retrieval_scores = evaluate(retrieved, truth)
    segments = evaluate_segments(predictions, truth, customer_segments(fitting, truth))
    catalog_shape = beyond_accuracy(
        predictions,
        dict(zip(counts["article_id"].to_list(), counts["n"].to_list())),
        dict(zip(catalog["article_id"].to_list(), catalog["product_type_no"].to_list())),
        catalog.height,
    )

    if save:
        save_result(MODEL_NAME, scores)
        write_report(scores, retrieval_scores, segments, catalog_shape, ceiling, pool_rows)

    return TwoStageResult(
        ranker=ranker, truth=truth, predictions=predictions, scores=scores, retrieval_scores=retrieval_scores,
        segments=segments, catalog_shape=catalog_shape, ceiling=ceiling, pool_rows=pool_rows,
        weeks_back=weeks_back, seed=seed,
    )


def main() -> dict:
    result = evaluate_two_stage(save=True)
    scores, pool_rows = result.scores, result.pool_rows
    print(f"{MODEL_NAME}: pool {pool_rows:,} pairs for {scores['n_customers']:,} customers "
          f"({pool_rows / scores['n_customers']:.0f} candidates each)")
    print(f"  pool ceiling (oracle recall): {result.ceiling:.4f}")
    print("  top features: " + ", ".join(f"{row[0]}={int(row[1])}" for row in result.importance().head(5).iter_rows()))
    print_scores(scores)

    retrieval = result.retrieval_scores
    print(f"  stage 1 (pool order):  map@12={retrieval['map@12']:.5f}  recall@12={retrieval['recall@12']:.5f}")
    print(f"  stage 2 (reranked):    map@12={scores['map@12']:.5f}  recall@12={scores['recall@12']:.5f}")
    for name, part in result.segments.items():
        print(f"  {name:<18} n={part['n_customers']:>5,}  map@12={part['map@12']:.5f}  recall@100={part['recall@100']:.5f}")
    print("  catalog shape: " + "  ".join(f"{k}={v:.4f}" for k, v in result.catalog_shape.items()))
    return scores


if __name__ == "__main__":
    main()
