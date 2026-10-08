"""Cache one Marqo-FashionSigLIP embedding per catalog article's description.

The companion of ``src/data/image_embeddings.py`` for the ``detail_desc`` sentence
H&M writes for every article. Preprocessing, not a model: the text tower runs
inference only. It is 768-dim, trained on fashion product text alongside the
images, and reads a description the way a shopper would - "jumper" and "sweater"
land together - where TF-IDF only matches the exact words. Descriptions longer
than its 64-token context are truncated; that affects 4% of them.

Every catalog article with a non-empty description gets a row; the rest are
left out and read back as zero vectors. Colour variants of a garment share one
description, so they share one embedding - this block says what an article is,
and the image block says what it looks like.

Embeddings are L2-normalised and written to ``data/text_embeddings_fashion_siglip.parquet``
with columns ``article_id`` and ``embedding``; an existing file is left alone unless
``--force`` is passed. Run it with ``python -m src.data.text_embeddings``.
"""

from __future__ import annotations

import sys

import open_clip
import polars as pl
import torch

from src.data.image_embeddings import FASHION_SIGLIP
from src.data.loading import catalog, load_articles
from src.paths import TEXT_EMBEDDINGS

BATCH_SIZE = 512


def catalog_descriptions() -> tuple[list[int], list[str]]:
    """Every catalog article that has a description, in article order."""
    article_ids = catalog()
    described = (
        pl.DataFrame({"article_id": article_ids})
        .join(load_articles().select("article_id", "detail_desc"), on="article_id", how="left")
        .filter(pl.col("detail_desc").is_not_null() & (pl.col("detail_desc").str.strip_chars() != ""))
    )
    print(f"{described.height:,} catalog articles with descriptions ({len(article_ids) - described.height:,} without)")
    return described["article_id"].to_list(), described["detail_desc"].to_list()


def extract(force: bool = False) -> pl.DataFrame:
    out = TEXT_EMBEDDINGS
    if out.exists() and not force:
        print(f"{out.name} already cached, skipping")
        return pl.read_parquet(out)

    article_ids, texts = catalog_descriptions()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(FASHION_SIGLIP)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(FASHION_SIGLIP)

    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), BATCH_SIZE):
            tokens = tokenizer(texts[start : start + BATCH_SIZE]).to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                batch = model.encode_text(tokens)
            embeddings.append(torch.nn.functional.normalize(batch.float(), dim=1).cpu())

    embeddings = torch.cat(embeddings)
    dim = embeddings.shape[1]
    assert embeddings.shape[0] == len(article_ids), embeddings.shape
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(len(article_ids)), atol=1e-3)

    table = pl.DataFrame(
        {"article_id": article_ids, "embedding": embeddings.tolist()},
        schema={"article_id": pl.Int64, "embedding": pl.Array(pl.Float32, dim)},
    )
    table.write_parquet(out)
    print(f"wrote {out.name} ({table.height:,} x {dim}, {out.stat().st_size / 1e6:.0f} MB)")
    return table


if __name__ == "__main__":
    extract(force="--force" in sys.argv)
