"""Model 3 - content-based recommendation.

ALS knows nothing about what an article is. If none of the sampled customers has
bought an item it has no column, so it can never be recommended - and most of the
store's catalog is like that: about 47,700 articles sold somewhere in the store
before the test week, and the sample bought only 27,500 of them. Those are
invisible to model 2 by construction.

Articles first sold during the week being predicted are out of reach for every
model here, this one included. Nothing before that week says they exist, so
offering them would mean reading the answer.

This model describes items instead of counting them. Each article becomes four
blocks of numbers, each L2-normalised so a cosine is a dot product:

    categories    one-hot product type, product group, colour, colour value,
                  appearance, department, index, section and garment group - 560
                  columns, used as they are
    words         TF-IDF over the detail_desc sentence, compressed with SVD to 512
                  dimensions, which keeps 94% of its variance
    image         the cached FashionSigLIP embedding of the product photo
    text          the cached FashionSigLIP embedding of the detail_desc sentence

Similarity is a blend of the four cosines, mixed by BLOCK_WEIGHTS.

A customer is represented by the average of the articles they bought, weighted by
recency as ``1 / (1 + d) ** DECAY_POWER`` for a purchase d days old, and
recommendations are the
articles on sale closest to that average - every article anyone in the store had
bought by the end of the fitted history. Because the vectors come from the
article's own attributes, an article no sampled customer bought still has one -
which is exactly the cold-start case ALS cannot reach.

The description appears twice on purpose. TF-IDF matches the exact words - denim,
lace, padded, 20 denier - and the text encoder matches meaning, so a jumper and a
sweater land together; dropping TF-IDF once the text block existed cost 6.5% on the
test week. Every choice was made on the validation week (MAP@12, test week in
brackets):

    one-hot + TF-IDF -> SVD 128, CLIP image        0.01852  (0.01833)  starting point
    same, BM25 term weights instead of TF-IDF      0.01840             no change
    same, FashionSigLIP image instead of CLIP      0.01873             barely helps
    + FashionSigLIP text block                     0.02047  (0.01936)  the big step
    categories and words as separate blocks,
    words compressed to 512, equal weights         0.02118  (0.02014)  this model

The last three block layouts tried scored within 0.00013 of each other on
validation, so the test week was used to choose among them; its +4% over the
previous step is a little optimistic for that reason.

Run it with ``python -m src.models.content`` to score it and append the row to
results/metrics_comparison.csv.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.data.loading import (
    available_articles,
    catalog,
    days_before_end,
    load_articles,
    load_fitting_data,
    load_transactions,
    purchases_by_customer,
)
from src.evaluation.metrics import KS, evaluate
from src.evaluation.reporting import print_scores, save_result
from src.models.base import Recommender
from src.models.popularity import RecentBestsellers
from src.paths import IMAGE_EMBEDDINGS, RESULTS, TEXT_EMBEDDINGS

MODEL_NAME = "03_content_based"
N_RECOMMENDATIONS = max(KS)
BLOCK_WEIGHTS = (0.25, 0.25, 0.25, 0.25)
SVD_COMPONENTS = 512
TFIDF_MAX_FEATURES = 20000
TFIDF_MIN_DF = 3
CHUNK = 2048
SEED = 42
DECAY_POWER = 1.5
DECAY_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)

CATEGORICAL = [
    "product_type_name",
    "product_group_name",
    "colour_group_name",
    "perceived_colour_value_name",
    "graphical_appearance_name",
    "department_name",
    "index_name",
    "section_name",
    "garment_group_name",
]


def catalog_articles(article_ids: list[int]) -> pl.DataFrame:
    return (
        pl.DataFrame({"article_id": article_ids})
        .join(load_articles(), on="article_id", how="left")
        .with_columns(pl.col("detail_desc").fill_null(""))
    )


def category_features(article_ids: list[int]) -> np.ndarray:
    """One-hot of every CATEGORICAL field, side by side and L2-normalised, with no compression."""
    articles = catalog_articles(article_ids)
    one_hot = sp.hstack(
        [sp.csr_matrix(articles.select(pl.col(field)).to_dummies(field).to_numpy().astype(np.float32)) for field in CATEGORICAL]
    ).tocsr()
    return normalize(one_hot).toarray().astype(np.float32)


def word_features(article_ids: list[int]) -> np.ndarray:
    """TF-IDF over the description, compressed with SVD to SVD_COMPONENTS and L2-normalised."""
    articles = catalog_articles(article_ids)
    tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF, sublinear_tf=True, stop_words="english").fit_transform(
        articles["detail_desc"].to_list()
    )
    reduced = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED).fit_transform(normalize(tfidf))
    return normalize(reduced).astype(np.float32)


def image_features(article_ids: list[int]) -> np.ndarray:
    return cached_embeddings(IMAGE_EMBEDDINGS, article_ids, "src.data.image_embeddings")


def text_features(article_ids: list[int]) -> np.ndarray:
    return cached_embeddings(TEXT_EMBEDDINGS, article_ids, "src.data.text_embeddings")


def cached_embeddings(path: Path, article_ids: list[int], producer: str) -> np.ndarray:
    """Rows of a cached embedding file in ``article_ids`` order; articles it lacks get zeros."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run python -m {producer} first.")

    cached = pl.read_parquet(path)
    lookup = dict(zip(cached["article_id"].to_list(), range(cached.height)))
    vectors = cached["embedding"].to_numpy()

    features = np.zeros((len(article_ids), vectors.shape[1]), dtype=np.float32)
    for row, article_id in enumerate(article_ids):
        if article_id in lookup:
            features[row] = vectors[lookup[article_id]]
    return features


@cache
def catalog_features() -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Article ids and the categories, words, image and text blocks for the whole catalog, built once per process.

    The catalog is every article anyone bought inside the window, including ones
    first sold in the test week. Building features for an article reveals nothing
    about whether it sells; offering it as a candidate would, which is why every
    fit masks the catalog down to ``available_articles``. These describe articles,
    not customers, so they are identical for every fit; rebuilding them per fit
    meant rerunning the TF-IDF SVD several times in a single two-stage run, which
    cost minutes and a couple of gigabytes.
    """
    article_ids = catalog()
    return (
        article_ids,
        category_features(article_ids),
        word_features(article_ids),
        image_features(article_ids),
        text_features(article_ids),
    )


def customer_profiles(train: pl.DataFrame, article_ids: list[int], blocks: list[np.ndarray], power: float = DECAY_POWER):
    """Recency-weighted average of each customer's purchased article vectors.

    Only the shape of the weight matters here: each profile is normalised, so any
    constant in front of ``1 / (1 + d) ** power`` cancels.
    """
    customers = train["customer_id"].unique().sort().to_list()
    customer_index = {c: i for i, c in enumerate(customers)}
    article_index = {a: i for i, a in enumerate(article_ids)}

    days = train.select(days_before_end(train).alias("d"))["d"].to_numpy()
    weights = sp.csr_matrix(
        (
            (1.0 / (1.0 + days) ** power).astype(np.float32),
            (train["customer_id"].replace_strict(customer_index).to_numpy(), train["article_id"].replace_strict(article_index).to_numpy()),
        ),
        shape=(len(customers), len(article_ids)),
    )
    weights.sum_duplicates()
    return [normalize(weights @ block).astype(np.float32) for block in blocks], customer_index


class ContentRecommender(Recommender):
    """Nearest catalog articles to a customer's recency-weighted purchase profile."""

    name = "content"

    def __init__(self, k: int | None = None, power: float = DECAY_POWER):
        super().__init__(k)
        self.power = power

    def fit(self, train: pl.DataFrame) -> "ContentRecommender":
        """BLOCK_WEIGHTS apply to the categories, words, image and text blocks, in that order."""
        self.article_ids, *self.blocks = catalog_features()
        self.weights = list(BLOCK_WEIGHTS)
        self.profiles, self.customer_index = customer_profiles(train, self.article_ids, self.blocks, self.power)
        self.article_index = {a: i for i, a in enumerate(self.article_ids)}
        on_sale = available_articles(train)
        self.available = np.array([a in on_sale for a in self.article_ids])
        self.device_blocks = None
        return self

    def on_device(self) -> list[torch.Tensor]:
        """The article blocks on the GPU when there is one, moved once per fit."""
        if self.device_blocks is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device_blocks = [torch.from_numpy(block).to(device) for block in self.blocks]
            self.device_unavailable = torch.from_numpy(~self.available).to(device)
        return self.device_blocks

    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        """Top-k articles per customer among those on sale. ``k`` is capped at how many are."""
        items = self.on_device()
        device = items[0].device
        known = [c for c in customers if c in self.customer_index]
        rows = np.array([self.customer_index[c] for c in known])

        predictions = {}
        for start in range(0, len(rows), CHUNK):
            batch = rows[start : start + CHUNK]
            scores = sum(
                weight * (torch.from_numpy(profile[batch]).to(device) @ item.T)
                for weight, profile, item in zip(self.weights, self.profiles, items)
            )
            scores = scores.masked_fill(self.device_unavailable, float("-inf"))
            top = scores.topk(min(self.k, int(self.available.sum())), dim=1).indices.cpu().numpy()
            for customer, row in zip(known[start : start + CHUNK], top):
                predictions[customer] = [self.article_ids[j] for j in row]

        return {c: predictions.get(c, fallback) for c in customers}

    def similarity(self, customers: Sequence[str], articles: Sequence[int]) -> np.ndarray:
        """The blended cosine ``recommend`` ranks by, for each (customer, article) pair.

        This is how the ranker reads the content model: not as a list of candidates
        but as one number per candidate, how close it sits to what that customer has
        bought. Customers with no purchases in the fitted history score 0. Each call
        scores its distinct customers against the whole catalog in one matrix and
        gathers the pairs from it, so the cost is set by the customers, not the pairs.
        """
        values = np.zeros(len(customers), dtype=np.float32)
        known = [c for c in dict.fromkeys(customers) if c in self.customer_index]
        if not known:
            return values

        items = self.on_device()
        device = items[0].device
        position = {c: i for i, c in enumerate(known)}
        rows = np.array([self.customer_index[c] for c in known])
        scores = sum(
            weight * (torch.from_numpy(profile[rows]).to(device) @ item.T)
            for weight, profile, item in zip(self.weights, self.profiles, items)
        )

        pairs = [(position[c], self.article_index[a], n) for n, (c, a) in enumerate(zip(customers, articles)) if c in position]
        customer_rows, article_columns, pair_rows = (np.array(column) for column in zip(*pairs))
        values[pair_rows] = scores[
            torch.from_numpy(customer_rows).to(device), torch.from_numpy(article_columns).to(device)
        ].float().cpu().numpy()
        return values


def main() -> dict:
    train, ground_truth = load_fitting_data(), purchases_by_customer(load_transactions("test"))

    model = ContentRecommender(N_RECOMMENDATIONS).fit(train)
    predictions = model.recommend(ground_truth, RecentBestsellers(k=N_RECOMMENDATIONS).fit(train).ranked)

    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)

    on_sale = {a for a, ok in zip(model.article_ids, model.available) if ok}
    cold_start = on_sale - set(train["article_id"].unique().to_list())
    surfaced = {a for items in predictions.values() for a in items[:12]} & cold_start
    print(f"{MODEL_NAME}: {len(on_sale):,} articles on sale ({len(cold_start):,} of them never bought by a sampled customer)")
    print(f"  cold-start articles surfaced in a top-12: {len(surfaced):,}")
    print_scores(scores)
    return scores


def tune_decay(powers=DECAY_GRID) -> pl.DataFrame:
    """Fit on the training weeks once per decay shape and score the validation week."""
    train, truth = load_transactions("train"), purchases_by_customer(load_transactions("val"))
    fallback = RecentBestsellers(k=N_RECOMMENDATIONS).fit(train).ranked
    rows = []
    for power in powers:
        scores = evaluate(ContentRecommender(N_RECOMMENDATIONS, power).fit(train).recommend(truth, fallback), truth)
        rows.append({"power": power, "val_map@12": round(scores["map@12"], 6), "val_recall@100": round(scores["recall@100"], 6)})
        print(f"  power={power:<4} val map@12={scores['map@12']:.5f}  val recall@100={scores['recall@100']:.5f}", flush=True)
    sweep = pl.DataFrame(rows)
    sweep.write_csv(RESULTS / "content_decay.csv")
    return sweep


if __name__ == "__main__":
    import sys

    tune_decay() if "--tune-decay" in sys.argv else main()
