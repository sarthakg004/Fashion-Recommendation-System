"""Model 3 - content-based recommendation.

ALS knows nothing about what an article is. If nobody has bought an item yet it
has no vector, so it can never be recommended - and this catalog turns over
constantly: 639 of the articles bought in the test week, 12% of them, had never
sold before that week. Those are invisible to model 2 by construction.

This model describes items instead of counting them. Each article becomes a
vector built from three sources:

    categorical   product type, colour, department, index, section, garment group
    text          TF-IDF over the detail_desc sentence
    image         the cached frozen-encoder embedding of the product photo

The categorical and text blocks are sparse and highly correlated, so they are
concatenated and reduced with SVD to a dense block. Similarity is then a blend of
two cosines, metadata and image, mixed by IMAGE_WEIGHT.

A customer is represented by the average of the articles they bought, weighted by
recency the same way model 2 weights its matrix, and recommendations are the
catalog articles closest to that average. Because the vectors come from the
article's own attributes, an article bought by nobody still has one - which is
exactly the cold-start case ALS cannot reach.

IMAGE_WEIGHT and IMAGE_ENCODER were chosen on the validation week; see the
notebook for the sweep.

Run it directly to score it and append the row to results/metrics_comparison.csv.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import load_articles, load_fitting_data, load_transactions, purchases_by_customer
from src.metrics import KS, evaluate, save_result

MODEL_NAME = "03_content_based"
N_RECOMMENDATIONS = max(KS)
IMAGE_ENCODER = "clip"
IMAGE_WEIGHT = 0.5
SVD_COMPONENTS = 128
TFIDF_MAX_FEATURES = 20000
TFIDF_MIN_DF = 3
CHUNK = 2048
SEED = 42

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


def catalog(transactions: pl.DataFrame) -> list[int]:
    """Every article seen anywhere in the sample, including test-only ones."""
    return sorted(transactions["article_id"].unique().to_list())


def metadata_features(article_ids: list[int]) -> np.ndarray:
    articles = (
        pl.DataFrame({"article_id": article_ids})
        .join(load_articles(), on="article_id", how="left")
        .with_columns(pl.col("detail_desc").fill_null(""))
    )

    one_hot = sp.hstack(
        [sp.csr_matrix(articles.select(pl.col(field)).to_dummies(field).to_numpy().astype(np.float32)) for field in CATEGORICAL]
    ).tocsr()
    text = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF, sublinear_tf=True, stop_words="english").fit_transform(
        articles["detail_desc"].to_list()
    )

    combined = sp.hstack([normalize(one_hot), normalize(text)]).tocsr()
    reduced = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=SEED).fit_transform(combined)
    return normalize(reduced).astype(np.float32)


def image_features(article_ids: list[int], encoder: str = IMAGE_ENCODER) -> np.ndarray:
    path = Path(__file__).resolve().parents[1] / "data" / f"image_embeddings_{encoder}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run data/extract_image_embeddings.py first.")

    cached = pl.read_parquet(path)
    lookup = dict(zip(cached["article_id"].to_list(), range(cached.height)))
    vectors = cached["embedding"].to_numpy()

    features = np.zeros((len(article_ids), vectors.shape[1]), dtype=np.float32)
    for row, article_id in enumerate(article_ids):
        if article_id in lookup:
            features[row] = vectors[lookup[article_id]]
    return features


def customer_profiles(train: pl.DataFrame, article_ids: list[int], blocks: list[np.ndarray]):
    """Recency-weighted average of each customer's purchased article vectors."""
    customers = train["customer_id"].unique().sort().to_list()
    customer_index = {c: i for i, c in enumerate(customers)}
    article_index = {a: i for i, a in enumerate(article_ids)}

    days = train.select(((pl.lit(train["t_dat"].max()) - pl.col("t_dat")).dt.total_days()).alias("d"))["d"].to_numpy()
    weights = sp.csr_matrix(
        (
            (1.0 / (1.0 + days)).astype(np.float32),
            (train["customer_id"].replace_strict(customer_index).to_numpy(), train["article_id"].replace_strict(article_index).to_numpy()),
        ),
        shape=(len(customers), len(article_ids)),
    )
    weights.sum_duplicates()
    return [normalize(weights @ block).astype(np.float32) for block in blocks], customer_index


def recommend(profiles, item_blocks, weights, customers, customer_index, article_ids, fallback,
              n: int = N_RECOMMENDATIONS) -> dict[str, list[int]]:
    """Top-n articles per customer. ``n`` is capped at the catalog size."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = [torch.from_numpy(block).to(device) for block in item_blocks]
    known = [c for c in customers if c in customer_index]
    rows = np.array([customer_index[c] for c in known])

    predictions = {}
    for start in range(0, len(rows), CHUNK):
        batch = rows[start : start + CHUNK]
        scores = sum(
            weight * (torch.from_numpy(profile[batch]).to(device) @ item.T)
            for weight, profile, item in zip(weights, profiles, items)
        )
        top = scores.topk(min(n, len(article_ids)), dim=1).indices.cpu().numpy()
        for customer, row in zip(known[start : start + CHUNK], top):
            predictions[customer] = [article_ids[j] for j in row]

    return {c: predictions.get(c, fallback) for c in customers}


def main(image_encoder: str = IMAGE_ENCODER, image_weight: float = IMAGE_WEIGHT, split: str = "test") -> dict:
    train, held_out = (load_fitting_data() if split == "test" else load_transactions("train")), load_transactions(split)
    ground_truth = purchases_by_customer(held_out)

    article_ids = catalog(load_transactions())
    blocks = [metadata_features(article_ids), image_features(article_ids, image_encoder)]
    profiles, customer_index = customer_profiles(train, article_ids, blocks)

    popularity = importlib.import_module("01_popularity")
    predictions = recommend(
        profiles, blocks, [1.0 - image_weight, image_weight], ground_truth, customer_index, article_ids, popularity.top_articles(train)
    )

    scores = evaluate(predictions, ground_truth)
    if split == "test":
        save_result(MODEL_NAME, scores)

    cold_start = set(article_ids) - set(train["article_id"].unique().to_list())
    surfaced = {a for items in predictions.values() for a in items[:12]} & cold_start
    print(f"{MODEL_NAME}: {len(article_ids):,} candidate articles ({len(cold_start):,} never sold before the test week)")
    print(f"  cold-start articles surfaced in a top-12: {len(surfaced):,}")
    for k in KS:
        print(f"  @{k:<4} " + "  ".join(f"{m}={scores[f'{m}@{k}']:.5f}" for m in ("precision", "recall", "hitrate", "ndcg", "map")))
    return scores


if __name__ == "__main__":
    main()
