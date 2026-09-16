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

Everything below was chosen on the validation week:

    temperature 0.15   at 0.05 the softmax is so peaked that a handful of
                       negatives dominate each step (val MAP@12 0.0129 -> 0.0157).
    learned item vector   adds the collaborative signal content alone cannot carry,
                       but only at the lower learning rate; at 1e-3 it memorises
                       instead (training loss 7.2 -> 5.1 while validation got worse).
    batch 4096         batch size sets how many in-batch negatives each step sees;
                       2048 and 16384 both scored lower.
    logQ correction    in-batch negatives are drawn in proportion to popularity, so
                       popular articles are punished for being popular. Subtracting
                       log(item frequency) from the logits undoes that sampling
                       bias and is worth about 8% (0.0156 -> 0.0168 at matched size).
    1024/256 towers    the model was capacity-starved rather than overfitting.
                       Averaged over three seeds, 1024/256 scores 0.01739 against
                       0.01688 for 512/256, a gap about three times the run-to-run
                       spread. Narrower still (128/64) was clearly worse, and more
                       dropout hurt at every width tried.
    early stopping     patience of 10 on the validation week, which settles between
                       epochs 11 and 20; the old fixed 20-epoch schedule was cutting
                       training off mid-improvement.

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

from src.data_utils import SAMPLE, load_fitting_data, load_transactions, purchases_by_customer
from src.metrics import KS, evaluate, save_result

content = importlib.import_module("03_content_based")
popularity = importlib.import_module("01_popularity")

MODEL_NAME = "04_two_tower"
N_RECOMMENDATIONS = max(KS)
EMBED_DIM = 256
HIDDEN_DIM = 1024
BATCH_SIZE = 4096
EPOCHS = 80
PATIENCE = 10
RETRIEVER_EPOCHS = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
USE_ITEM_EMBEDDING = True
USE_LOGQ_CORRECTION = True
TEMPERATURE = 0.15
DROPOUT = 0.1
CHUNK = 4096
SEED = 42
ARTIFACTS = Path(__file__).resolve().parent / "artifacts" / "two_tower"

CUSTOMER_CATEGORICAL = ["club_member_status", "fashion_news_frequency"]


_ITEM_FEATURES: tuple[list[int], np.ndarray] | None = None


def item_feature_matrix() -> tuple[list[int], np.ndarray]:
    """Catalog features for every article, built once per process.

    These describe articles, not customers, so they are identical for every fit.
    Rebuilding them per fit meant rerunning the metadata SVD five times in a
    single two-stage run, which cost minutes and a couple of gigabytes.
    """
    global _ITEM_FEATURES
    if _ITEM_FEATURES is None:
        article_ids = content.catalog(load_transactions())
        features = np.hstack(
            [
                (1 - content.IMAGE_WEIGHT) * content.metadata_features(article_ids),
                content.IMAGE_WEIGHT * content.image_features(article_ids, content.IMAGE_ENCODER),
            ]
        ).astype(np.float32)
        _ITEM_FEATURES = (article_ids, features)
    return _ITEM_FEATURES


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

    def __init__(self, in_dim: int, n_items: int, use_item_embedding: bool = USE_ITEM_EMBEDDING,
                 hidden: int = HIDDEN_DIM, out_dim: int = EMBED_DIM):
        super().__init__()
        self.content = Tower(in_dim, hidden, out_dim)
        self.embedding = nn.Embedding(n_items, out_dim) if use_item_embedding else None
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


def rank(user_tower, item_tower, item_features, customers, customer_index, article_ids, history_sum, history_weight,
         static, fallback, device, n: int = N_RECOMMENDATIONS):
    """Top ``n`` articles per customer.

    ``n`` defaults to the largest k the metrics need, but a candidate retriever
    asks for far more than that - the pool is judged on coverage, not on the
    twelve it would have shown.
    """
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
            top = (user_vectors @ item_vectors.T).topk(min(n, len(article_ids)), dim=1).indices.cpu().numpy()
            for customer, row in zip(known[start : start + CHUNK], top):
                predictions[customer] = [article_ids[j] for j in row]

    return {c: predictions.get(c, fallback) for c in customers}


def train_model(verbose: bool = True, epochs: int | None = None, learning_rate: float = LEARNING_RATE,
                temperature: float = TEMPERATURE, use_item_embedding: bool = USE_ITEM_EMBEDDING,
                batch_size: int = BATCH_SIZE, transactions: pl.DataFrame | None = None,
                select_best: bool = True, use_logq: bool = USE_LOGQ_CORRECTION, patience: int = PATIENCE,
                save_artifacts: bool = False, hidden_dim: int = HIDDEN_DIM, embed_dim: int = EMBED_DIM):
    """Train both towers.

    ``transactions`` overrides the training frame, which matters when this is used
    as a candidate retriever for an earlier week: the model must see only that
    week's history, never the week it is nominating candidates for.

    ``select_best`` controls how training stops. With it on, the validation week
    scores every epoch, training stops after ``patience`` epochs without
    improvement, and the best weights are restored. With it off there is no
    honest validation week available - it lies in the future of an earlier
    origin - so training runs a fixed ``RETRIEVER_EPOCHS``, the count early
    stopping settled on during tuning.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = load_transactions("train") if transactions is None else transactions
    article_ids, item_features = item_feature_matrix()

    customers, customer_index, rows, cols, weights, history_sum, history_weight = build_training_data(train, article_ids, item_features)
    static = customer_features(customers)

    epochs = epochs if epochs is not None else (EPOCHS if select_best else RETRIEVER_EPOCHS)

    user_tower = Tower(item_features.shape[1] + static.shape[1], hidden_dim, embed_dim).to(device)
    item_tower = ItemTower(item_features.shape[1], len(article_ids), use_item_embedding, hidden_dim, embed_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(user_tower.parameters()) + list(item_tower.parameters()), lr=learning_rate, weight_decay=WEIGHT_DECAY
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    counts = np.bincount(cols, weights=weights, minlength=len(article_ids)).astype(np.float32)
    log_q = torch.from_numpy(np.log(np.clip(counts / counts.sum(), 1e-9, None))).to(device)

    val_truth = purchases_by_customer(load_transactions("val")) if select_best else {}
    fallback = popularity.top_articles(train)
    best_map, best_state, waited, history = -1.0, None, 0, []

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
                if use_logq:
                    logits = logits - log_q[item_ids].unsqueeze(0)
                loss = F.cross_entropy(logits, torch.arange(len(batch), device=device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(batch)

        schedule.step()
        epoch_loss = total_loss / len(order)

        if not select_best:
            history.append({"epoch": epoch, "loss": epoch_loss, "val_map@12": None})
            if verbose:
                print(f"  epoch {epoch:>2}  loss={epoch_loss:.4f}")
            continue

        predictions = rank(user_tower, item_tower, item_features, val_truth, customer_index, article_ids, history_sum, history_weight, static, fallback, device)
        val_map = evaluate(predictions, val_truth)["map@12"]
        history.append({"epoch": epoch, "loss": epoch_loss, "val_map@12": val_map})

        improved = val_map > best_map
        if improved:
            best_map, waited = val_map, 0
            best_state = ({k: v.clone() for k, v in user_tower.state_dict().items()}, {k: v.clone() for k, v in item_tower.state_dict().items()})
        else:
            waited += 1
        if verbose:
            print(f"  epoch {epoch:>2}  loss={epoch_loss:.4f}  val_map@12={val_map:.5f}{'  *' if improved else ''}")
        if waited >= patience:
            if verbose:
                print(f"  early stop: no gain for {patience} epochs, best was epoch {max(history, key=lambda h: h['val_map@12'] or -1)['epoch']}")
            break

    if best_state is not None:
        user_tower.load_state_dict(best_state[0])
        item_tower.load_state_dict(best_state[1])
    if save_artifacts:
        write_artifacts(history)
    if device == "cuda":
        torch.cuda.empty_cache()
    return dict(
        user_tower=user_tower, item_tower=item_tower, item_features=item_features, article_ids=article_ids,
        customer_index=customer_index, history_sum=history_sum, history_weight=history_weight, static=static,
        fallback=fallback, device=device, best_val_map=best_map if select_best else None, history=history,
    )


def write_artifacts(history: list[dict]) -> None:
    """Save the training history and its curve next to the model.

    The figure is built through the Figure API rather than pyplot, because
    pyplot would need a backend and switching to a file-writing one changes it
    for the whole process. Called from a notebook, that silently stops every
    later plot from displaying.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(history)
    frame.write_parquet(ARTIFACTS / "history.parquet")

    scored = frame.filter(pl.col("val_map@12").is_not_null())
    fig = Figure(figsize=(8, 3.4))
    FigureCanvasAgg(fig)
    left = fig.add_subplot(111)
    left.plot(frame["epoch"], frame["loss"], color="#4c1fb8", linewidth=1.6, label="training loss")
    left.set_xlabel("epoch")
    left.set_ylabel("training loss", color="#4c1fb8")
    left.spines[["top"]].set_visible(False)

    if scored.height:
        right = left.twinx()
        right.plot(scored["epoch"], scored["val_map@12"], color="#0f9d76", linewidth=1.6, label="val MAP@12")
        best = scored.sort("val_map@12", descending=True).head(1)
        right.scatter(best["epoch"], best["val_map@12"], color="#0f9d76", zorder=3)
        right.annotate(f"best epoch {best['epoch'][0]}", (best["epoch"][0], best["val_map@12"][0]),
                       textcoords="offset points", xytext=(6, -10), fontsize=8, color="#0f9d76")
        right.set_ylabel("val MAP@12", color="#0f9d76")
        right.spines[["top"]].set_visible(False)

    left.set_title("Two-tower training", fontsize=10)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "training_curve.png", dpi=150)


def main() -> dict:
    """Tune on the validation week, then refit on train+val and score the test week.

    The first pass fits on training data only and early-stops on the validation
    week, which is what produces the saved training curve. The second refits on
    everything before the test week for the epoch count the first pass chose,
    because by then the validation week is training data and cannot referee
    anything.
    """
    tuned = train_model(transactions=load_transactions("train"), select_best=True, save_artifacts=True)
    chosen = max(tuned["history"], key=lambda h: h["val_map@12"] or -1)["epoch"]
    print(f"{MODEL_NAME}: early stopping chose epoch {chosen} (val map@12={tuned['best_val_map']:.5f}), refitting on train+val")

    trained = train_model(transactions=load_fitting_data(), select_best=False, epochs=chosen, verbose=False)
    ground_truth = purchases_by_customer(load_transactions("test"))
    predictions = rank(
        trained["user_tower"], trained["item_tower"], trained["item_features"], ground_truth, trained["customer_index"],
        trained["article_ids"], trained["history_sum"], trained["history_weight"], trained["static"], trained["fallback"], trained["device"],
    )

    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)
    print(f"{MODEL_NAME}: artifacts in {ARTIFACTS.relative_to(ARTIFACTS.parents[2])}")
    for k in KS:
        print(f"  @{k:<4} " + "  ".join(f"{m}={scores[f'{m}@{k}']:.5f}" for m in ("precision", "recall", "hitrate", "ndcg", "map")))
    return scores


if __name__ == "__main__":
    main()
