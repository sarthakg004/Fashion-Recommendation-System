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
learned retriever of the five. Earlier versions pooled all six classes in
``src/retrievers.py`` at 500 candidates each and reached a recall ceiling of 0.32;
these two at 700 reach about 0.47 on a third of the code, because the other four
were mostly nominating articles these two already had. The rest are kept because
they stay cheap to pool back in, not because the default needs them.

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
from nine to seventeen moved test MAP@12 by about 9%, and four of the five
strongest features by gain are among the new ones.

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

One week scored once is a single draw. ``src/validation.py`` re-runs this whole
file over several held-out weeks and several seeds, which is what separates a
real gain from the pipeline's own noise - and that noise is not negligible, since
the two-tower is retrained per fold on a GPU and two identical runs land about
0.0005 MAP@12 apart.

Run it directly to build, train, score and append the row to
results/metrics_comparison.csv.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import datetime as dt
import gc
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMRanker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_utils import load_articles, load_fitting_data, purchases_by_customer, split_at
from src.metrics import KS, beyond_accuracy, evaluate, evaluate_segments, save_result
from src.retrievers import Retriever, default_retrievers

MODEL_NAME = "05_two_stage_ranker"
REPORT_PATH = Path(__file__).resolve().parents[1] / "results" / "two_stage_report.json"
N_RECOMMENDATIONS = max(KS)
MISSING_RANK = 9999
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
    """How much an article sells, how recently, and whether it is on the way up.

    ``article_velocity`` is last week's sales over the week before. A coat selling
    three times as fast as it did last week is a different proposition from one
    with the same total that is fading, and total counts cannot express that.
    """
    last_day = train["t_dat"].max()
    recent = train.filter(pl.col("t_dat") > last_day - dt.timedelta(days=7))
    previous = train.filter(
        (pl.col("t_dat") <= last_day - dt.timedelta(days=7)) & (pl.col("t_dat") > last_day - dt.timedelta(days=14))
    )

    stats = train.group_by("article_id").agg(
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


def customer_statistics(train: pl.DataFrame) -> pl.DataFrame:
    """How much a customer buys, how recently, and what they usually spend."""
    last_day = train["t_dat"].max()
    return train.group_by("customer_id").agg(
        pl.len().alias("customer_purchases"),
        ((pl.lit(last_day) - pl.col("t_dat").max()).dt.total_days()).alias("customer_days_since_purchase"),
        pl.col("price").mean().alias("customer_price"),
        pl.col("t_dat").n_unique().alias("customer_active_days"),
    )


def affinity_statistics(train: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """What a customer has bought before, at the product and category level.

    The exact article is the narrowest possible match. A customer who bought a
    garment in black is a strong candidate for the same garment in green, which
    shares a ``product_code``, and a weaker but real one for anything else of that
    product type.
    """
    catalog = load_articles().select("article_id", "product_code", "product_type_no")
    tagged = train.join(catalog, on="article_id", how="left")
    by_code = tagged.group_by("customer_id", "product_code").agg(pl.len().alias("product_bought_before"))
    by_type = tagged.group_by("customer_id", "product_type_no").agg(pl.len().alias("type_bought_before"))
    return by_code, by_type


def add_features(pool: pl.DataFrame, train: pl.DataFrame, articles: pl.DataFrame, customers: pl.DataFrame,
                 rank_columns: list[str]) -> pl.DataFrame:
    repurchase = train.group_by("customer_id", "article_id").agg(pl.len().alias("times_bought_before"))
    by_code, by_type = affinity_statistics(train)
    catalog = load_articles().select("article_id", "product_code", "product_type_no")

    return (
        pool.with_columns(
            pl.sum_horizontal([pl.col(column).is_not_null().cast(pl.Int32) for column in rank_columns]).alias("n_sources"),
            pl.min_horizontal([pl.col(column) for column in rank_columns]).fill_null(MISSING_RANK).cast(pl.Int32).alias("best_rank"),
        )
        .with_columns([pl.col(column).fill_null(MISSING_RANK) for column in rank_columns])
        .join(articles, on="article_id", how="left")
        .join(customers, on="customer_id", how="left")
        .join(repurchase, on=["customer_id", "article_id"], how="left")
        .join(catalog, on="article_id", how="left")
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


def rank_pool(model: LGBMRanker, pool: pl.DataFrame, customers, features: list[str], fallback: list[int]) -> dict[str, list[int]]:
    scored = pool.with_columns(pl.Series("score", model.predict(pool.select(features).to_numpy())))
    return take_top(scored, "score", customers, fallback, descending=True)


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


def label_weeks(n_weeks: int = LABEL_WEEKS, fitting: pl.DataFrame | None = None):
    """Rolling origin: each week paired with the history available before it.

    ``fitting`` is everything the model may learn from, so its final week is the
    most recent week that can be labelled. Defaults to the fixed train-plus-val
    frame; cross-validation passes an earlier one.
    """
    labelled = load_fitting_data() if fitting is None else fitting
    last_day = labelled["t_dat"].max()

    for index in range(n_weeks):
        end = last_day - dt.timedelta(days=7 * index)
        start = end - dt.timedelta(days=6)
        week = labelled.filter((pl.col("t_dat") >= start) & (pl.col("t_dat") <= end))
        history = labelled.filter(pl.col("t_dat") < start)
        yield index, history, purchases_by_customer(week)


def build_training_pool(n_weeks: int = LABEL_WEEKS, verbose: bool = True,
                        fitting: pl.DataFrame | None = None, seed: int = SEED) -> pl.DataFrame:
    """Stack several labelled weeks, each built from its own history.

    Two things keep this inside memory. Negatives are dropped before the feature
    joins, because all but a few percent of a week's pool is thrown away and
    joining statistics onto the full thing first is wasted work. And customers
    are processed in batches, so peak memory depends on CUSTOMER_BATCH rather
    than on how many candidates each retriever nominates.
    """
    weeks = []
    for index, history, truth in label_weeks(n_weeks, fitting):
        retrievers = [r.fit(history) for r in default_retrievers()]
        rank_columns = [f"rank_{r.name}" for r in retrievers]
        bestsellers = retrievers[0].recommend(["_"])["_"][:N_RECOMMENDATIONS]
        articles, customers = article_statistics(history, bestsellers), customer_statistics(history)

        batches, raw_pairs, positives = [], 0, 0
        for batch in batched(list(truth), CUSTOMER_BATCH):
            candidates = label(build_pool(retrievers, batch), truth)
            raw_pairs += candidates.height
            positives += int(candidates["bought"].sum())
            batches.append(add_features(downsample(candidates, seed=seed), history, articles, customers, rank_columns))
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


def write_report(scores, retrieval_scores, segments, catalog_shape, ceiling, pool_rows,
                 path: Path = REPORT_PATH) -> Path:
    """Everything about the final model that the README quotes, as one file.

    The README used to be written by hand from whatever run happened to be on
    screen, which is how it twice ended up disagreeing with the results file.
    Anything the README states about this model is regenerated from here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {
            "ceiling": round(ceiling, 5),
            "pool_rows": pool_rows,
            "candidates_per_customer": round(pool_rows / scores["n_customers"]),
            "final": {k: round(v, 6) if isinstance(v, float) else v for k, v in scores.items()},
            "retrieval": {k: round(v, 6) if isinstance(v, float) else v for k, v in retrieval_scores.items()},
            "segments": {name: {k: round(v, 6) if isinstance(v, float) else v for k, v in part.items()}
                         for name, part in segments.items()},
            "catalog_shape": {k: round(v, 5) for k, v in catalog_shape.items()},
        },
        indent=2,
    ))
    return path


def run(retrievers: list[Retriever] | None = None, n_weeks: int = LABEL_WEEKS, verbose: bool = True,
        save: bool = True, weeks_back: int = 0, seed: int = SEED,
        window_weeks: int | None = None) -> dict:
    """Fit both stages and return everything the notebook wants to look at.

    ``weeks_back`` chooses which week to hold out, counting back from the test
    week, and ``seed`` drives both the negative sample and the ranker. Varying
    them is how the cross-validation in ``src/validation.py`` separates a real
    difference from the noise of one week and one random draw. ``window_weeks``
    caps how much history is used, which is what holds training size constant
    while the evaluation week moves.
    """
    retrievers = retrievers or default_retrievers()
    fitting, evaluation = split_at(weeks_back, window_weeks)
    test_truth = purchases_by_customer(evaluation)

    train_pool = build_training_pool(n_weeks, verbose, fitting=fitting, seed=seed)

    for retriever in retrievers:
        retriever.fit(fitting)

    features = feature_names(retrievers)
    rank_columns = [f"rank_{r.name}" for r in retrievers]
    bestsellers = retrievers[0].recommend(["_"])["_"][:N_RECOMMENDATIONS]
    articles, customers = article_statistics(fitting, bestsellers), customer_statistics(fitting)
    pool_rows = 0

    groups = train_pool.group_by("week", "customer_id", maintain_order=True).len()["len"].to_numpy()
    model = LGBMRanker(**{**LGBM_PARAMS, "random_state": seed})
    model.fit(train_pool.select(features).to_numpy(), train_pool["bought"].to_numpy(), group=groups)

    predictions, retrieved, reachable = {}, {}, []
    for batch in batched(list(test_truth), CUSTOMER_BATCH):
        pool = build_pool(retrievers, batch)
        reachable.append(oracle_recall(pool, {c: test_truth[c] for c in batch}))
        featured = add_features(pool, fitting, articles, customers, rank_columns)
        predictions.update(rank_pool(model, featured, batch, features, bestsellers))
        retrieved.update(take_top(featured, "best_rank", batch, bestsellers, descending=False))
        pool_rows += featured.height
        del pool, featured
        gc.collect()

    ceiling = float(np.mean(reachable))
    scores = evaluate(predictions, test_truth)

    counts = fitting.group_by("article_id").agg(pl.len().alias("n"))
    catalog = load_articles().select("article_id", "product_type_no")
    retrieval_scores = evaluate(retrieved, test_truth)
    segments = evaluate_segments(predictions, test_truth, customer_segments(fitting, test_truth))
    catalog_shape = beyond_accuracy(
        predictions,
        dict(zip(counts["article_id"].to_list(), counts["n"].to_list())),
        dict(zip(catalog["article_id"].to_list(), catalog["product_type_no"].to_list())),
        catalog.height,
    )

    if save:
        save_result(MODEL_NAME, scores)
        write_report(scores, retrieval_scores, segments, catalog_shape, ceiling, pool_rows)

    return dict(
        scores=scores, predictions=predictions, model=model, train_pool=train_pool, pool_rows=pool_rows,
        ceiling=ceiling, bestsellers=bestsellers, retrievers=retrievers, weeks_back=weeks_back, seed=seed,
        retrieval_scores=retrieval_scores, segments=segments, catalog_shape=catalog_shape,
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

    retrieval = result["retrieval_scores"]
    print(f"  stage 1 (pool order):  map@12={retrieval['map@12']:.5f}  recall@12={retrieval['recall@12']:.5f}")
    print(f"  stage 2 (reranked):    map@12={scores['map@12']:.5f}  recall@12={scores['recall@12']:.5f}")
    for name, part in result["segments"].items():
        print(f"  {name:<18} n={part['n_customers']:>5,}  map@12={part['map@12']:.5f}  recall@100={part['recall@100']:.5f}")
    print("  catalog shape: " + "  ".join(f"{k}={v:.4f}" for k, v in result["catalog_shape"].items()))
    return scores


if __name__ == "__main__":
    main()
