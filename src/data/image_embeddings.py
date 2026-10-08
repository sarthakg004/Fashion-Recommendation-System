"""Cache one Marqo-FashionSigLIP embedding per sampled article photo.

This is preprocessing, not a model: the encoder runs inference only, no
fine-tuning. Marqo-FashionSigLIP is a CLIP-style pair of encoders (ViT-B/16 SigLIP)
trained on fashion product images and their text; this module caches its image
tower, 768-dim, and ``src/data/text_embeddings.py`` caches its text tower. ResNet-18
and a general-purpose CLIP were compared against it and are documented in the
README; neither is used.

Embeddings are L2-normalised, so downstream cosine similarity is a plain dot
product. The photo loaders are spawned rather than forked: a forked worker starts
as a copy of a process already holding the model, and four of those ran out of
memory under a 3.5 GB cap. The output is ``data/image_embeddings_fashion_siglip.parquet`` with columns
``article_id`` and ``embedding``; an existing file is left alone unless ``--force``
is passed.

The cache covers every article anyone bought inside the window
(``src.data.loading.catalog``), not only the ones the sample bought, because any of
them can be recommended once it has sold. Re-run this whenever the sample is rebuilt. Run it with ``python -m src.data.image_embeddings``.
"""

from __future__ import annotations

import sys

import open_clip
import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.loading import catalog
from src.paths import IMAGE_EMBEDDINGS, RAW_IMAGES, image_path

BATCH_SIZE = 128
NUM_WORKERS = 4
MAX_MISSING_FRACTION = 0.05
FASHION_SIGLIP = "hf-hub:Marqo/marqo-fashionSigLIP"


class ArticleImages(Dataset):
    def __init__(self, article_ids, transform):
        self.article_ids = article_ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.article_ids)

    def __getitem__(self, i):
        article_id = self.article_ids[i]
        with Image.open(image_path(article_id)) as img:
            return self.transform(img.convert("RGB")), article_id


def catalog_with_photos() -> list[int]:
    article_ids = catalog()

    present = [a for a in article_ids if image_path(a).exists()]
    missing = len(article_ids) - len(present)
    if missing > MAX_MISSING_FRACTION * len(article_ids):
        raise RuntimeError(
            f"{missing:,} of {len(article_ids):,} catalog articles have no photo under {RAW_IMAGES}. "
            "That is more than expected - check the image folder layout."
        )
    print(f"{len(present):,} catalog articles with photos ({missing:,} without)")
    return present


def extract(force: bool = False) -> pl.DataFrame:
    out = IMAGE_EMBEDDINGS
    if out.exists() and not force:
        print(f"{out.name} already cached, skipping")
        return pl.read_parquet(out)

    article_ids = catalog_with_photos()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, transform = open_clip.create_model_and_transforms(FASHION_SIGLIP)
    model.eval().to(device)

    loader = DataLoader(
        ArticleImages(article_ids, transform),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        multiprocessing_context="spawn",
        pin_memory=device == "cuda",
    )

    embeddings, ids = [], []
    with torch.inference_mode():
        for batch, batch_ids in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                out_batch = model.encode_image(batch)
            embeddings.append(torch.nn.functional.normalize(out_batch.float(), dim=1).cpu())
            ids.append(batch_ids)
            print(f"  {sum(len(i) for i in ids):,} / {len(article_ids):,}", end="\r")

    embeddings = torch.cat(embeddings)
    dim = embeddings.shape[1]
    assert embeddings.shape[0] == len(article_ids), embeddings.shape
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(len(article_ids)), atol=1e-3)

    table = pl.DataFrame(
        {"article_id": torch.cat(ids).tolist(), "embedding": embeddings.tolist()},
        schema={"article_id": pl.Int64, "embedding": pl.Array(pl.Float32, dim)},
    )
    table.write_parquet(out)
    print(f"\nwrote {out.name} ({table.height:,} x {dim}, {out.stat().st_size / 1e6:.0f} MB)")
    return table


if __name__ == "__main__":
    extract(force="--force" in sys.argv)
