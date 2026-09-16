"""Read-only access to the artifacts written by precompute.py."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "raw" / "images"


@dataclass(frozen=True)
class Article:
    article_id: int
    name: str
    product_type: str
    colour: str
    section: str
    description: str


class RecommendationStore:
    """Everything the API serves, loaded once at startup.

    The whole dataset here is a few megabytes, so it stays in memory and every
    lookup is a dictionary hit rather than a query.
    """

    def __init__(self, artifacts: Path | None = None):
        self.artifacts = artifacts or Path(__file__).resolve().parent / "artifacts"
        if not (self.artifacts / "recommendations.parquet").exists():
            raise FileNotFoundError(f"No artifacts in {self.artifacts} - run python api/precompute.py first.")

    def _grouped(self, name: str) -> dict[str, list[int]]:
        frame = pl.read_parquet(self.artifacts / f"{name}.parquet")
        if "rank" in frame.columns:
            frame = frame.sort("rank")
        grouped = frame.group_by("customer_id", maintain_order=True).agg(pl.col("article_id"))
        return dict(zip(grouped["customer_id"].to_list(), grouped["article_id"].to_list()))

    @cached_property
    def recommendations(self) -> dict[str, list[int]]:
        return self._grouped("recommendations")

    @cached_property
    def purchases(self) -> dict[str, list[int]]:
        return self._grouped("purchases")

    @cached_property
    def history(self) -> dict[str, list[int]]:
        return self._grouped("history")

    @cached_property
    def articles(self) -> dict[int, Article]:
        frame = pl.read_parquet(self.artifacts / "articles.parquet")
        return {
            row["article_id"]: Article(
                article_id=row["article_id"],
                name=row["prod_name"] or "Unnamed",
                product_type=row["product_type_name"] or "Unknown",
                colour=row["colour_group_name"] or "Unknown",
                section=row["index_name"] or "Unknown",
                description=row["detail_desc"] or "",
            )
            for row in frame.iter_rows(named=True)
        }

    @cached_property
    def metrics(self) -> dict:
        import json

        return json.loads((self.artifacts / "metrics.json").read_text())

    @cached_property
    def customer_ids(self) -> list[str]:
        """Customers with the most interesting profiles first - most hits, then most bought."""
        ranked = sorted(
            self.recommendations,
            key=lambda c: (-self.hit_count(c), -len(self.purchases.get(c, [])), c),
        )
        return ranked

    def hit_count(self, customer_id: str) -> int:
        bought = set(self.purchases.get(customer_id, []))
        return sum(1 for a in self.recommendations.get(customer_id, []) if a in bought)

    def article(self, article_id: int) -> Article:
        return self.articles.get(
            article_id, Article(article_id, "Unknown", "Unknown", "Unknown", "Unknown", "")
        )

    def summaries(self, limit: int, query: str | None = None) -> list[dict]:
        ids = self.customer_ids
        if query:
            ids = [c for c in ids if c.startswith(query.lower())]
        return [
            {
                "customer_id": c,
                "bought": len(self.purchases.get(c, [])),
                "history": len(self.history.get(c, [])),
                "hits": self.hit_count(c),
            }
            for c in ids[:limit]
        ]

    def detail(self, customer_id: str) -> dict | None:
        if customer_id not in self.recommendations:
            return None
        bought = set(self.purchases.get(customer_id, []))

        def render(article_id: int, **extra) -> dict:
            article = self.article(article_id)
            return {
                "article_id": article.article_id,
                "name": article.name,
                "product_type": article.product_type,
                "colour": article.colour,
                "section": article.section,
                "image": f"/api/images/{article.article_id}",
                **extra,
            }

        return {
            "customer_id": customer_id,
            "hits": self.hit_count(customer_id),
            "recommendations": [
                render(a, rank=i, hit=a in bought)
                for i, a in enumerate(self.recommendations[customer_id], start=1)
            ],
            "history": [render(a) for a in self.history.get(customer_id, [])],
            "purchases": [
                render(a, hit=a in set(self.recommendations[customer_id]))
                for a in self.purchases.get(customer_id, [])
            ],
        }

    def image_path(self, article_id: int) -> Path | None:
        name = f"{article_id:010d}"
        path = IMAGES / name[:3] / f"{name}.jpg"
        return path if path.exists() else None
