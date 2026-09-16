"""Run the two-stage ranker once and cache what the API needs to serve.

Fitting the two-stage model takes several minutes - it refits six retrievers per
label week - so it happens here, once, and the API only ever reads parquet. The
recommendations served are exactly the ones scored in the comparison table; this
script does not re-rank anything, it just records the output.

Writes to ``api/artifacts/``:

    recommendations.parquet  top-12 per test customer, in rank order
    purchases.parquet        what each customer actually bought in the test week
    history.parquet          their most recent purchases before it
    articles.parquet         metadata for every article either file mentions
    metrics.json             the model's row from the comparison table
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))

from src.data_utils import load_articles, load_fitting_data, load_transactions, purchases_by_customer

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
TOP_K = 12
HISTORY_K = 8


def main() -> None:
    ranker = importlib.import_module("05_two_stage_ranker")
    result = ranker.run(save=False)
    predictions = result["predictions"]

    recommendations = pl.DataFrame(
        {
            "customer_id": [c for c, items in predictions.items() for _ in items[:TOP_K]],
            "article_id": [a for items in predictions.values() for a in items[:TOP_K]],
            "rank": [i for items in predictions.values() for i in range(1, len(items[:TOP_K]) + 1)],
        },
        schema={"customer_id": pl.String, "article_id": pl.Int64, "rank": pl.Int32},
    )

    test = load_transactions("test")
    purchases = test.select("customer_id", "article_id").unique()

    history = (
        load_fitting_data()
        .sort("t_dat", descending=True)
        .unique(subset=["customer_id", "article_id"], keep="first", maintain_order=True)
        .filter(pl.col("customer_id").is_in(recommendations["customer_id"].unique()))
        .group_by("customer_id", maintain_order=True)
        .agg(pl.col("article_id").head(HISTORY_K))
        .explode("article_id")
    )

    mentioned = set(recommendations["article_id"]) | set(purchases["article_id"]) | set(history["article_id"])
    articles = load_articles().filter(pl.col("article_id").is_in(list(mentioned))).select(
        "article_id", "prod_name", "product_type_name", "colour_group_name", "index_name", "detail_desc"
    )

    scores = {"model": ranker.MODEL_NAME, **{k: round(v, 6) if isinstance(v, float) else v for k, v in result["scores"].items()}}

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    recommendations.write_parquet(ARTIFACTS / "recommendations.parquet")
    purchases.write_parquet(ARTIFACTS / "purchases.parquet")
    history.write_parquet(ARTIFACTS / "history.parquet")
    articles.write_parquet(ARTIFACTS / "articles.parquet")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(scores, indent=2))

    print(f"customers       {recommendations['customer_id'].n_unique():,}")
    print(f"recommendations {recommendations.height:,}")
    print(f"articles        {articles.height:,}")
    print(f"map@12          {scores['map@12']:.5f}")


if __name__ == "__main__":
    main()
