"""The one interface every recommender in this project implements.

A model is fitted on a frame of transactions and then asked for a ranked list of
articles per customer. That is the whole contract, and it is what lets the same
class be scored on its own as one of the five models and pooled as a candidate
source by the two-stage ranker without a wrapper in between.

Customers a model knows nothing about get ``fallback``. Scored on its own, a model
passes the popularity list so nobody is left empty-handed. As a retriever it
passes nothing, because the bestseller retriever already covers everyone and
duplicating it would waste pool slots.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import polars as pl


class Recommender(ABC):
    """Fitted on transactions, then asked for the top ``k`` articles per customer."""

    name: str = "recommender"
    default_k: int = 200

    def __init__(self, k: int | None = None):
        self.k = self.default_k if k is None else k

    @abstractmethod
    def fit(self, train: pl.DataFrame) -> "Recommender":
        ...

    @abstractmethod
    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(k={self.k})"
