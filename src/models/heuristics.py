"""Cheap rule-based candidate sources for the two-stage model.

Neither of these is a model in its own right, and neither is in the default pool
today, but both are high precision per candidate and cost almost nothing to pool
back in. Measured recall on the test week:

    PreviousPurchases      0.020 from about 8 candidates - the highest precision
                           per candidate of anything tried
    ColourVariants         0.043 - customers rebuy the same garment in a
                           different colour, and those share a product_code

They were dropped from the default pool because they saturate: a customer has
only about eight previous purchases to re-offer and about fifty colour variants of
what they bought, so they stop contributing as the pool grows, while the
bestseller and two-tower retrievers keep paying as k rises.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

from src.data.loading import load_articles
from src.models.base import Recommender


class PreviousPurchases(Recommender):
    """The customer's own articles, most recently bought first."""

    name = "repurchase"
    default_k = 50

    def __init__(self, k: int | None = None):
        super().__init__(k)
        self.history: dict[str, list[int]] = {}

    def fit(self, train: pl.DataFrame) -> "PreviousPurchases":
        grouped = (
            train.sort("t_dat", descending=True)
            .unique(subset=["customer_id", "article_id"], keep="first", maintain_order=True)
            .group_by("customer_id", maintain_order=True)
            .agg(pl.col("article_id").head(self.k))
        )
        self.history = dict(zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list()))
        return self

    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        return {customer: self.history.get(customer, []) for customer in customers}


class ColourVariants(Recommender):
    """Other colourways of garments the customer already bought.

    H&M gives every colourway of the same garment a shared ``product_code``, so
    this is a one-join retriever and it finds purchases no similarity model does.
    Variants are ordered by how much they sell, so truncating at k keeps the
    plausible ones.
    """

    name = "variant"
    default_k = 100

    def __init__(self, k: int | None = None):
        super().__init__(k)
        self.variants: dict[str, list[int]] = {}

    def fit(self, train: pl.DataFrame) -> "ColourVariants":
        articles = load_articles().select("article_id", "product_code")
        popularity = train.group_by("article_id").agg(pl.len().alias("purchases"))

        catalog = (
            articles.join(popularity, on="article_id", how="left")
            .with_columns(pl.col("purchases").fill_null(0))
            .sort("purchases", descending=True)
        )
        by_code = catalog.group_by("product_code").agg(pl.col("article_id"))
        code_to_articles = dict(zip(by_code["product_code"].to_list(), by_code["article_id"].to_list()))

        bought_codes = (
            train.join(articles, on="article_id", how="left")
            .group_by("customer_id")
            .agg(pl.col("product_code").unique())
        )
        self.variants = {
            customer: [a for code in codes for a in code_to_articles.get(code, [])][: self.k]
            for customer, codes in zip(bought_codes["customer_id"].to_list(), bought_codes["product_code"].to_list())
        }
        return self

    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        return {customer: self.variants.get(customer, []) for customer in customers}
