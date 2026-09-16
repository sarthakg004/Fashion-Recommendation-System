"""Cache one image embedding per sampled article, using frozen pretrained encoders.

This is preprocessing, not a model: both encoders run inference only, no
fine-tuning. Two are cached side by side so models 3 and 4 can be tried on either
without re-running anything:

    resnet18  ImageNet-supervised ResNet-18, classifier head removed, 512-dim.
              Cheap, and its similarity is dominated by shape and colour.
    clip      OpenCLIP ViT-B/32 (LAION-2B), image tower only, 512-dim. Trained on
              image-text pairs, so its similarity tracks how a garment would be
              described, not only how it looks.

Embeddings are L2-normalised, so downstream cosine similarity is a plain dot
product. Outputs are written per encoder to
``data/image_embeddings_{name}.parquet`` with columns ``article_id`` and
``embedding``; an existing file is left alone unless ``FORCE`` is set.

The cache is keyed to the article list in ``data/sample``, so re-run this whenever
the sample is rebuilt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

BATCH_SIZE = 256
NUM_WORKERS = 8
MAX_MISSING_FRACTION = 0.05
CLIP_ARCH = "ViT-B-32"
CLIP_WEIGHTS = "laion2b_s34b_b79k"

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "raw" / "images"
SAMPLE = ROOT / "data" / "sample" / "transactions.parquet"


def output_path(encoder: str) -> Path:
    return ROOT / "data" / f"image_embeddings_{encoder}.parquet"


def image_path(article_id: int) -> Path:
    name = f"{article_id:010d}"
    return IMAGES / name[:3] / f"{name}.jpg"


def load_resnet18(device: str):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, weights.transforms()


def load_clip(device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(CLIP_ARCH, pretrained=CLIP_WEIGHTS)
    model.eval().to(device)
    return model.encode_image, preprocess


ENCODERS = {"resnet18": load_resnet18, "clip": load_clip}


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


def sampled_articles() -> list[int]:
    if not SAMPLE.exists():
        raise FileNotFoundError(f"{SAMPLE} not found - run data/sample_data.py first.")
    article_ids = sorted(pl.read_parquet(SAMPLE, columns=["article_id"])["article_id"].unique().to_list())

    present = [a for a in article_ids if image_path(a).exists()]
    missing = len(article_ids) - len(present)
    if missing > MAX_MISSING_FRACTION * len(article_ids):
        raise RuntimeError(
            f"{missing:,} of {len(article_ids):,} sampled articles have no photo under {IMAGES}. "
            "That is more than expected - check the image folder layout."
        )
    print(f"{len(present):,} sampled articles with photos ({missing:,} without)")
    return present


def extract(encoder: str = "resnet18", force: bool = False) -> pl.DataFrame:
    out = output_path(encoder)
    if out.exists() and not force:
        print(f"{out.name} already cached, skipping")
        return pl.read_parquet(out)

    article_ids = sampled_articles()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encode, transform = ENCODERS[encoder](device)

    loader = DataLoader(
        ArticleImages(article_ids, transform),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device == "cuda",
    )

    embeddings, ids = [], []
    with torch.inference_mode():
        for batch, batch_ids in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                out_batch = encode(batch)
            embeddings.append(torch.nn.functional.normalize(out_batch.float(), dim=1).cpu())
            ids.append(batch_ids)
            print(f"  {encoder}: {sum(len(i) for i in ids):,} / {len(article_ids):,}", end="\r")

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
    force = "--force" in sys.argv
    for name in ENCODERS:
        extract(name, force=force)
