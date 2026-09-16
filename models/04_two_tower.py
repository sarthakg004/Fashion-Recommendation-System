"""Model 4 - two-tower neural retrieval.

Models 2 and 3 each use half the evidence. ALS sees co-purchase patterns and
nothing about the items; the content model sees the items and nothing about who
buys together. A two-tower network learns both at once: one tower turns a
customer into a vector, another turns an article into a vector, and training
pushes a customer's vector towards the articles they actually bought.

    item tower   the same content features model 3 uses - SVD-reduced metadata
                 and the cached image embedding - plus a free per-article vector
                 learned from the interactions themselves. The learned part is
                 what carries the collaborative signal; it stays at its zero
                 initialisation for articles nobody bought, so those fall back to
                 pure content and cold-start survives.
    user tower   the recency-weighted average of the articles the customer
                 bought, plus their own attributes (age, club status, news
                 frequency), pushed through the same kind of MLP.

Training uses in-batch negatives: within a batch of (customer, bought article)
pairs, every other customer's article is treated as a negative, so one matrix
multiply supplies thousands of negatives for free. The loss is cross-entropy over
cosine similarities scaled by a temperature.

One subtlety that matters. The user vector is an average of purchased articles,
so if the positive article is left in that average the model can score it by
recognising itself, learning nothing. Each training pair therefore subtracts its
own article from the customer's average first - exact, and cheaper than
recomputing the average per sample.

Everything below was chosen on the validation week, best epoch kept:

    temperature 0.15   the single biggest lever. At 0.05 the softmax is so peaked
                       that a handful of negatives dominate each step
                       (val MAP@12 0.0129 -> 0.0157).
    learned item vector   adds the collaborative signal content alone cannot carry
                       (0.0139 -> 0.0156 at matched settings), but only at the
                       lower learning rate; at 1e-3 it memorises instead
                       (training loss 7.2 -> 5.1 while validation got worse).
    batch 4096         batch size sets how many in-batch negatives each step sees;
                       2048 and 16384 both scored lower.

Run it directly to train, score and append the row to results/metrics_comparison.csv.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_utils import SAMPLE, load_transactions, purchases_by_customer
from src.metrics import KS, evaluate, save_result

content = importlib.import_module("03_content_based")
popularity = importlib.import_module("01_popularity")

MODEL_NAME = "04_two_tower"
N_RECOMMENDATIONS = max(KS)
EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 3e-4
USE_ITEM_EMBEDDING = True
TEMPERATURE = 0.15
DROPOUT = 0.1
CHUNK = 4096
SEED = 42

CUSTOMER_CATEGORICAL = ["club_member_status", "fashion_news_frequency"]


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN_DIM, out_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ItemTower(nn.Module):
    """Content projection plus an optional learned per-article vector."""

    def __init__(self, in_dim: int, n_items: int, use_item_embedding: bool = USE_ITEM_EMBEDDING):
        super().__init__()
        self.content = Tower(in_dim)
        self.embedding = nn.Embedding(n_items, EMBED_DIM) if use_item_embedding else None
        if self.embedding is not None:
            nn.init.zeros_(self.embedding.weight)

    def forward(self, features, item_ids=None):
        vectors = self.content(features)
        if self.embedding is not None and item_ids is not None:
            vectors = vectors + self.embedding(item_ids)
        return F.normalize(vectors, dim=-1)


def customer_features(customers: list[str]) -> np.ndarray:
    table = (
        pl.DataFrame({"customer_id": customers})
        .join(pl.read_parquet(SAMPLE / "customers.parquet"), on="customer_id", how="left")
        .with_columns(
            ((pl.col("age") - pl.col("age").median()) / pl.col("age").std()).fill_null(0.0).alias("age_z"),
            pl.col("FN").fill_null(0.0),
            pl.col("Active").fill_null(0.0),
            *[pl.col(field).fill_null("unknown") for field in CUSTOMER_CATEGORICAL],
        )
    )
    dummies = table.select(CUSTOMER_CATEGORICAL).to_dummies(CUSTOMER_CATEGORICAL)
    return np.hstack([table.select("age_z", "FN", "Active").to_numpy(), dummies.to_numpy()]).astype(np.float32)


def build_training_data(train: pl.DataFrame, article_ids: list[int], item_features: np.ndarray):
    """Aggregated (customer, article) pairs with recency weights, and the history sums."""
    customers = train["customer_id"].unique().sort().to_list()
    customer_index = {c: i for i, c in enumerate(customers)}
    article_index = {a: i for i, a in enumerate(article_ids)}

    pairs = (
        train.with_columns(((pl.lit(train["t_dat"].max()) - pl.col("t_dat")).dt.total_days()).alias("days"))
        .with_columns((1.0 / (1.0 + pl.col("days"))).alias("weight"))
        .group_by("customer_id", "article_id")
        .agg(pl.col("weight").sum())
    )

    rows = pairs["customer_id"].replace_strict(customer_index).to_numpy()
    cols = pairs["article_id"].replace_strict(article_index).to_numpy()
    weights = pairs["weight"].to_numpy().astype(np.float32)

    history_sum = np.zeros((len(customers), item_features.shape[1]), dtype=np.float32)
    np.add.at(history_sum, rows, weights[:, None] * item_features[cols])
    history_weight = np.zeros(len(customers), dtype=np.float32)
    np.add.at(history_weight, rows, weights)

    return customers, customer_index, rows, cols, weights, history_sum, history_weight


def user_inputs(history_sum, history_weight, static, rows, exclude_cols=None, exclude_weights=None, item_features=None):
    totals = history_sum[rows]
    denominators = history_weight[rows]
    if exclude_cols is not None:
        totals = totals - exclude_weights[:, None] * item_features[exclude_cols]
        denominators = denominators - exclude_weights
    profile = totals / np.clip(denominators, 1e-6, None)[:, None]
    return np.hstack([profile, static[rows]]).astype(np.float32)


def rank(user_tower, item_tower, item_features, customers, customer_index, article_ids, history_sum, history_weight, static, fallback, device):
    user_tower.eval()
    item_tower.eval()
    with torch.inference_mode():
        item_vectors = torch.cat(
            [
                item_tower(
                    torch.from_numpy(item_features[i : i + CHUNK]).to(device),
                    torch.arange(i, min(i + CHUNK, len(item_features)), device=device),
                )
                for i in range(0, len(item_features), CHUNK)
            ]
        )
        known = [c for c in customers if c in customer_index]
        rows = np.array([customer_index[c] for c in known])

        predictions = {}
        for start in range(0, len(rows), CHUNK):
            batch = rows[start : start + CHUNK]
            features = user_inputs(history_sum, history_weight, static, batch)
            user_vectors = user_tower(torch.from_numpy(features).to(device))
            top = (user_vectors @ item_vectors.T).topk(N_RECOMMENDATIONS, dim=1).indices.cpu().numpy()
            for customer, row in zip(known[start : start + CHUNK], top):
                predictions[customer] = [article_ids[j] for j in row]

    return {c: predictions.get(c, fallback) for c in customers}


def train_model(verbose: bool = True, epochs: int = EPOCHS, learning_rate: float = LEARNING_RATE,
                temperature: float = TEMPERATURE, use_item_embedding: bool = USE_ITEM_EMBEDDING,
                batch_size: int = BATCH_SIZE):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = load_transactions("train")
    article_ids = content.catalog(load_transactions())
    item_features = np.hstack(
        [
            (1 - content.IMAGE_WEIGHT) * content.metadata_features(article_ids),
            content.IMAGE_WEIGHT * content.image_features(article_ids, content.IMAGE_ENCODER),
        ]
    ).astype(np.float32)

    customers, customer_index, rows, cols, weights, history_sum, history_weight = build_training_data(train, article_ids, item_features)
    static = customer_features(customers)

    user_tower = Tower(item_features.shape[1] + static.shape[1]).to(device)
    item_tower = ItemTower(item_features.shape[1], len(article_ids), use_item_embedding).to(device)
    optimizer = torch.optim.AdamW(list(user_tower.parameters()) + list(item_tower.parameters()), lr=learning_rate)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    val_truth = purchases_by_customer(load_transactions("val"))
    fallback = popularity.top_articles(train)
    best_map, best_state = -1.0, None

    for epoch in range(1, epochs + 1):
        user_tower.train()
        item_tower.train()
        order = np.random.permutation(len(rows))
        total_loss = 0.0

        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            if len(batch) < 2:
                continue
            features = user_inputs(history_sum, history_weight, static, rows[batch], cols[batch], weights[batch], item_features)
            users = torch.from_numpy(features).to(device)
            items = torch.from_numpy(item_features[cols[batch]]).to(device)
            item_ids = torch.from_numpy(cols[batch]).to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                logits = (user_tower(users) @ item_tower(items, item_ids).T) / temperature
                loss = F.cross_entropy(logits, torch.arange(len(batch), device=device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(batch)

        predictions = rank(user_tower, item_tower, item_features, val_truth, customer_index, article_ids, history_sum, history_weight, static, fallback, device)
        val_map = evaluate(predictions, val_truth)["map@12"]
        if val_map > best_map:
            best_map = val_map
            best_state = ({k: v.clone() for k, v in user_tower.state_dict().items()}, {k: v.clone() for k, v in item_tower.state_dict().items()})
        if verbose:
            print(f"  epoch {epoch:>2}  loss={total_loss / len(order):.4f}  val_map@12={val_map:.5f}{'  *' if val_map == best_map else ''}")

    user_tower.load_state_dict(best_state[0])
    item_tower.load_state_dict(best_state[1])
    return dict(
        user_tower=user_tower, item_tower=item_tower, item_features=item_features, article_ids=article_ids,
        customer_index=customer_index, history_sum=history_sum, history_weight=history_weight, static=static,
        fallback=fallback, device=device, best_val_map=best_map,
    )


def main() -> dict:
    trained = train_model()
    ground_truth = purchases_by_customer(load_transactions("test"))
    predictions = rank(
        trained["user_tower"], trained["item_tower"], trained["item_features"], ground_truth, trained["customer_index"],
        trained["article_ids"], trained["history_sum"], trained["history_weight"], trained["static"], trained["fallback"], trained["device"],
    )

    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)
    print(f"{MODEL_NAME}: best val map@12={trained['best_val_map']:.5f}")
    for k in KS:
        print(f"  @{k:<4} " + "  ".join(f"{m}={scores[f'{m}@{k}']:.5f}" for m in ("precision", "recall", "hitrate", "ndcg", "map")))
    return scores


if __name__ == "__main__":
    main()
