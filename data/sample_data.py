"""Sample the H&M dataset down to something a single machine iterates on quickly.

The full transactions file is 3.3 GB / 31.8 M rows over two years, which is too
slow to loop over while building five models. Fashion is seasonal and the catalog
turns over weekly, so the window comes first and everything else is derived from
it:

1. Filter transactions to the last ``WINDOW_DAYS`` days. Nothing downstream ever
   sees a row outside this window - a customer's 2018 purchases are not part of
   this dataset even if that customer is sampled, because a two-year-old catalog
   is not a signal about what sells next week.
2. Sample ``CUSTOMER_FRACTION`` of the customers active inside that window, and
   keep all of their in-window transactions. Customers are sampled rather than
   transactions so purchase sequences survive intact for models 3-5. The customer
   list is sorted before sampling because ``unique`` does not return a stable row
   order, and an unsorted seed would hand back a different sample on every run -
   which would silently invalidate the cached image embeddings keyed to it.
3. Split the sampled frame by time: last week test, second-to-last week val,
   the rest of the window train.

``articles.csv`` is kept whole (105 K rows, small). ``customers.csv`` is filtered
to the sampled customers. The split is written as a ``split`` column so
downstream scripts filter instead of recomputing dates.

Outputs to ``data/sample/``: transactions.parquet, customers.parquet,
articles.parquet.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

WINDOW_DAYS = 140
CUSTOMER_FRACTION = 0.06
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sample"
SOURCES = ("transactions_train.csv", "customers.csv", "articles.csv")


def check_download() -> None:
    missing = [name for name in SOURCES if not (RAW / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {RAW}. Download the H&M Personalized Fashion "
            "Recommendations dataset from Kaggle and unzip it there."
        )
    if not (RAW / "images").is_dir():
        raise FileNotFoundError(f"Missing image folder at {RAW / 'images'}.")


def build() -> pl.DataFrame:
    transactions = pl.scan_csv(RAW / "transactions_train.csv", schema_overrides={"t_dat": pl.Date})

    last_day = transactions.select(pl.col("t_dat").max()).collect(engine="streaming").item()
    window_start = last_day - dt.timedelta(days=WINDOW_DAYS - 1)
    val_start = last_day - dt.timedelta(days=13)
    test_start = last_day - dt.timedelta(days=6)

    window = transactions.filter(pl.col("t_dat") >= window_start)
    sampled = (
        window.select("customer_id")
        .unique()
        .sort("customer_id")
        .collect(engine="streaming")
        .sample(fraction=CUSTOMER_FRACTION, seed=SEED)
    )

    sample = (
        window.join(sampled.lazy(), on="customer_id", how="semi")
        .with_columns(
            pl.when(pl.col("t_dat") >= test_start)
            .then(pl.lit("test"))
            .when(pl.col("t_dat") >= val_start)
            .then(pl.lit("val"))
            .otherwise(pl.lit("train"))
            .alias("split")
        )
        .sort("customer_id", "t_dat")
        .collect(engine="streaming")
    )

    OUT.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(OUT / "transactions.parquet")
    (
        pl.scan_csv(RAW / "customers.csv")
        .join(sampled.lazy(), on="customer_id", how="semi")
        .collect(engine="streaming")
        .write_parquet(OUT / "customers.parquet")
    )
    pl.read_csv(RAW / "articles.csv").write_parquet(OUT / "articles.parquet")

    print(f"window        {window_start} .. {last_day} ({WINDOW_DAYS} days)")
    print(f"train         {window_start} .. {val_start - dt.timedelta(days=1)}")
    print(f"val           {val_start} .. {test_start - dt.timedelta(days=1)}")
    print(f"test          {test_start} .. {last_day}")
    print(f"customers     {sample['customer_id'].n_unique():,}")
    print(f"articles      {sample['article_id'].n_unique():,}")
    print(f"transactions  {sample.height:,}")
    print(f"date range    {sample['t_dat'].min()} .. {sample['t_dat'].max()}")
    return sample


if __name__ == "__main__":
    check_download()
    build()
