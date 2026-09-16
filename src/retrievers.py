"""Stage-one candidate retrievers for the two-stage model.

Every retriever answers the same question - "which articles are worth scoring for
this customer?" - and they are deliberately allowed to be bad at it. Stage two
sorts the pool; stage one only has to make sure the right article is somewhere in
it. What matters here is coverage, so the cheap heuristics below sit alongside the
trained models rather than being replaced by them.

Measured recall on the test week, which is why each one exists:

    RecentBestsellers      0.178 at k=300 - what is selling right now beats any
                           model trained on a four-month average
    ColourVariants         0.043 - customers rebuy the same garment in a
                           different colour, and those share a product_code
    PreviousPurchases      0.020 from about 8 candidates - the highest precision
                           per candidate of anything here
    AlsRetriever           collaborative signal
    ContentRetriever       reaches articles nobody has bought yet
    TwoTowerRetriever      learned blend of the two

All six together reach far more than any one of them: the pool built from this
list is what sets the ceiling the ranker works against.

Each retriever is fitted once on the training frame and then asked for candidates
for whatever customers it is given. Unknown customers get an empty list rather
than a popularity fallback, because the bestseller retriever already covers
everyone and duplicating it would waste pool slots.
"""

from __future__ import annotations

import importlib
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def _model(name: str):
    for path in (ROOT, ROOT / "models"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module(name)


class Retriever(ABC):
    """One candidate source. Fit on train, then asked for top-k per customer."""

    name: str = "retriever"
    default_k: int = 200

    def __init__(self, k: int | None = None):
        self.k = self.default_k if k is None else k

    @abstractmethod
    def fit(self, train: pl.DataFrame) -> "Retriever":
        ...

    @abstractmethod
    def recommend(self, customers: list[str]) -> dict[str, list[int]]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(k={self.k})"


class RecentBestsellers(Retriever):
    """What sold most in the final days of training, same list for everyone."""

    name = "recent"
    default_k = 300

    def __init__(self, days: int = 7, k: int | None = None):
        super().__init__(k)
        self.days = days
        self.ranked: list[int] = []

    def fit(self, train: pl.DataFrame) -> "RecentBestsellers":
        cutoff = train["t_dat"].max() - pl.duration(days=self.days)
        self.ranked = (
            train.filter(pl.col("t_dat") > cutoff)
            .group_by("article_id")
            .agg(pl.len().alias("purchases"))
            .sort("purchases", descending=True)
            .head(self.k)["article_id"]
            .to_list()
        )
        return self

    def recommend(self, customers: list[str]) -> dict[str, list[int]]:
        return {customer: self.ranked for customer in customers}


class PreviousPurchases(Retriever):
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

    def recommend(self, customers: list[str]) -> dict[str, list[int]]:
        return {customer: self.history.get(customer, []) for customer in customers}


class ColourVariants(Retriever):
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
        articles = pl.read_parquet(ROOT / "data" / "sample" / "articles.parquet").select("article_id", "product_code")
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

    def recommend(self, customers: list[str]) -> dict[str, list[int]]:
        return {customer: self.variants.get(customer, []) for customer in customers}


class AlsRetriever(Retriever):
    """Model 2, wrapped as a candidate source."""

    name = "als"

    def fit(self, train: pl.DataFrame) -> "AlsRetriever":
        als = _model("02_collaborative_als")
        matrix, self.items, self.index = als.build_matrix(train)
        self.model, self.weighted = als.fit(matrix)
        self._als = als
        return self

    def recommend(self, customers: list[str]) -> dict[str, list[int]]:
        ranked = self._als.recommend(self.model, self.weighted, customers, self.items, self.index, [])
        return {customer: items[: self.k] for customer, items in ranked.items()}


class ContentRetriever(Retriever):
    """Model 3, wrapped as a candidate source."""

    name = "content"

    def fit(self, train: pl.DataFrame) -> "ContentRetriever":
        content = _model("03_content_based")
        from src.data_utils import load_transactions

        self.articles = content.catalog(load_transactions())
        self.blocks = [
            content.metadata_features(self.articles),
            content.image_features(self.articles, content.IMAGE_ENCODER),
        ]
        self.profiles, self.index = content.customer_profiles(train, self.articles, self.blocks)
        self.weights = [1 - content.IMAGE_WEIGHT, content.IMAGE_WEIGHT]
        self._content = content
        return self

    def recommend(self, customers: list[str]) -> dict[str, list[int]]:
        ranked = self._content.recommend(
            self.profiles, self.blocks, self.weights, customers, self.index, self.articles, []
        )
        return {customer: items[: self.k] for customer, items in ranked.items()}


class TwoTowerRetriever(Retriever):
    """Model 4, wrapped as a candidate source."""

    name = "tower"

    def fit(self, train: pl.DataFrame) -> "TwoTowerRetriever":
        tower = _model("04_two_tower")
        self.trained = tower.train_model(verbose=False, transactions=train, select_best=False)
        self._tower = tower
        return self

    def recommend(self, customers: list[str]) -> dict[str, list[int]]:
        trained = self.trained
        ranked = self._tower.rank(
            trained["user_tower"], trained["item_tower"], trained["item_features"], customers,
            trained["customer_index"], trained["article_ids"], trained["history_sum"],
            trained["history_weight"], trained["static"], [], trained["device"],
        )
        return {customer: items[: self.k] for customer, items in ranked.items()}


def default_retrievers() -> list[Retriever]:
    return [
        RecentBestsellers(),
        PreviousPurchases(),
        ColourVariants(),
        AlsRetriever(),
        ContentRetriever(),
        TwoTowerRetriever(),
    ]
