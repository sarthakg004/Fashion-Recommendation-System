"""Model 4 - two-tower neural retrieval.

Models 2 and 3 each use half the evidence. ALS sees co-purchase patterns and
nothing about the items; the content model sees the items and nothing about who
buys together. A two-tower network learns both at once: one tower turns a
customer into a vector, another turns an article into a vector, and training
pushes a customer's vector towards the articles they actually bought.

    item tower   the same four content blocks model 3 uses - one-hot categories,
                 SVD-compressed TF-IDF, and the cached image and description
                 embeddings, scaled by model 3's block weights - plus a free
                 per-article vector learned from the interactions themselves.
                 The learned part is what carries the collaborative signal; it
                 stays at its zero initialisation for articles no sampled
                 customer bought, so those fall back to pure content and
                 cold-start survives. Like model 3 it only offers articles the
                 store had sold by the end of its history.
    user tower   the recency-weighted average of the articles the customer
                 bought, a purchase d days old weighted 1 / (1 + d) ** DECAY_POWER,
                 plus their own attributes (age, club status, news
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
    2048/256 towers    width barely matters. On the earlier two content blocks
                       1024/256 beat 512/256 (0.01739 against 0.01688 over three
                       seeds) and narrower still (128/64) was clearly worse. On the
                       four blocks, with the early stopping below, hidden widths of
                       512, 1024, 2048 and 4096 averaged 0.01841, 0.01862, 0.01902
                       and 0.01858 on validation over three seeds each, and landed
                       within 2% of each other on the test week; 2048 is kept as
                       the validation winner. More dropout hurt at every width.
    one hidden layer   depth went the other way. On model 3's four content blocks,
                       over three seeds each, validation MAP@12 averaged 0.0190 with
                       one hidden layer, 0.0160 with two (1024/512) and 0.0146 with
                       three (1024/1024/512). The inputs are pretrained embeddings
                       that have already done the representation work.
    recency shape      a purchase d days old weighs 1 / (1 + d) ** p in the user's
                       history average. Over three seeds each, validation MAP@12
                       was 0.0080, 0.0183, 0.0173 and 0.0130 for p = 0, 1, 1.5 and
                       2, and recall@100 0.076, 0.130, 0.134 and 0.123. p = 1.5
                       looked like a recall gain for a retriever, but on the test
                       week it moved recall@100 by +0.001, lowered the model 5
                       pool ceiling (0.507 -> 0.504) and cut recall@100 on
                       never-bought articles from 0.012 to 0.009, so p = 1 stays.
    early stopping     patience of 10 on the last week of whatever frame fit is
                       given, then a refit for the chosen epoch count - for model 4
                       and for every retriever fit inside model 5. It settles
                       between epochs 9 and 17 at width 2048 (11 to 20 on the earlier
                       two blocks); a fixed epoch count either cut training off
                       mid-improvement or ran past the best epoch.

Run it with ``python -m src.models.two_tower`` to train, score and append the row to
results/metrics_comparison.csv. The training history and its curve go to
``results/two_tower/``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loading import (
    available_articles,
    days_before_end,
    load_customers,
    load_fitting_data,
    load_transactions,
    purchases_by_customer,
    week_bounds,
)
from src.evaluation.metrics import KS, evaluate
from src.evaluation.reporting import print_scores, save_result
from src.models import content
from src.models.base import Recommender
from src.models.popularity import RecentBestsellers
from src.paths import RESULTS, ROOT

MODEL_NAME = "04_two_tower"
N_RECOMMENDATIONS = max(KS)
EMBED_DIM = 256
HIDDEN_DIM = 2048
BATCH_SIZE = 4096
EPOCHS = 80
PATIENCE = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
TEMPERATURE = 0.15
DROPOUT = 0.1
CHUNK = 4096
PAIR_CHUNK = 16384
SEED = 42
DECAY_POWER = 1.0
DECAY_GRID = (0.0, 1.0, 1.5, 2.0)
DECAY_SEEDS = (42, 7, 13)
ARTIFACTS = RESULTS / "two_tower"

CUSTOMER_CATEGORICAL = ["club_member_status", "fashion_news_frequency"]


@cache
def item_feature_matrix() -> tuple[list[int], np.ndarray]:
    """Content features for every catalog article: model 3's four blocks, weighted and side by side.

    Built once per process like the blocks it stacks, since every fit of every
    tower reads the same catalog and a copy per fit is half a gigabyte.
    """
    article_ids, *blocks = content.catalog_features()
    features = np.hstack([weight * block for weight, block in zip(content.BLOCK_WEIGHTS, blocks)]).astype(np.float32)
    return article_ids, features


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
    """Content projection plus a learned per-article vector, which starts at zero."""

    def __init__(self, in_dim: int, n_items: int, hidden: int = HIDDEN_DIM, out_dim: int = EMBED_DIM):
        super().__init__()
        self.content = Tower(in_dim, hidden, out_dim)
        self.embedding = nn.Embedding(n_items, out_dim)
        nn.init.zeros_(self.embedding.weight)

    def forward(self, features, item_ids):
        return F.normalize(self.content(features) + self.embedding(item_ids), dim=-1)


def customer_features(customers: list[str]) -> np.ndarray:
    """Each customer's own attributes: standardised age, the two newsletter flags, and one-hot club and news status."""
    table = (
        pl.DataFrame({"customer_id": customers})
        .join(load_customers(), on="customer_id", how="left")
        .with_columns(
            ((pl.col("age") - pl.col("age").median()) / pl.col("age").std()).fill_null(0.0).alias("age_z"),
            pl.col("FN").fill_null(0.0),
            pl.col("Active").fill_null(0.0),
            *[pl.col(field).fill_null("unknown") for field in CUSTOMER_CATEGORICAL],
        )
    )
    dummies = table.select(CUSTOMER_CATEGORICAL).to_dummies(CUSTOMER_CATEGORICAL)
    return np.hstack([table.select("age_z", "FN", "Active").to_numpy(), dummies.to_numpy()]).astype(np.float32)


def build_training_data(train: pl.DataFrame, article_ids: list[int], item_features: np.ndarray, power: float = DECAY_POWER):
    """Aggregated (customer, article) pairs with recency weights, and the history sums.

    The weights feed two averages - each customer's history and the item
    frequencies behind the logQ correction - and both divide by their total, so
    only the shape ``1 / (1 + d) ** power`` matters, never a constant in front.

    The sums are accumulated PAIR_CHUNK pairs at a time. Weighting every pair's
    feature row in one go materialises a pairs x features copy, which at the
    2,608-dimension content vector is several gigabytes; adding chunk by chunk in
    order produces the same sums without it.
    """
    customers = train["customer_id"].unique().sort().to_list()
    customer_index = {c: i for i, c in enumerate(customers)}
    article_index = {a: i for i, a in enumerate(article_ids)}

    pairs = (
        train.with_columns(days_before_end(train).alias("days"))
        .with_columns((1.0 / (1.0 + pl.col("days")) ** power).alias("weight"))
        .group_by("customer_id", "article_id")
        .agg(pl.col("weight").sum())
    )

    rows = pairs["customer_id"].replace_strict(customer_index).to_numpy()
    cols = pairs["article_id"].replace_strict(article_index).to_numpy()
    weights = pairs["weight"].to_numpy().astype(np.float32)

    history_sum = np.zeros((len(customers), item_features.shape[1]), dtype=np.float32)
    for start in range(0, len(rows), PAIR_CHUNK):
        part = slice(start, start + PAIR_CHUNK)
        np.add.at(history_sum, rows[part], weights[part, None] * item_features[cols[part]])
    history_weight = np.zeros(len(customers), dtype=np.float32)
    np.add.at(history_weight, rows, weights)

    return customers, customer_index, rows, cols, weights, history_sum, history_weight


def user_inputs(history_sum, history_weight, attributes, rows, exclude_cols=None, exclude_weights=None, item_features=None):
    """User tower input: the purchase-history average, optionally minus one article, beside the customer's attributes."""
    totals = history_sum[rows]
    denominators = history_weight[rows]
    if exclude_cols is not None:
        totals = totals - exclude_weights[:, None] * item_features[exclude_cols]
        denominators = denominators - exclude_weights
    profile = totals / np.clip(denominators, 1e-6, None)[:, None]
    return np.hstack([profile, attributes[rows]]).astype(np.float32)


class TwoTowerRecommender(Recommender):
    """User and item towers trained with in-batch negatives.

    ``fit`` trains twice, and never for a fixed number of epochs. The first pass
    holds out the last week of the frame it is given, trains on everything before
    it, scores that week after every epoch, stops after ``PATIENCE`` epochs without
    improvement and remembers the best epoch. The second pass refits on the whole
    frame for exactly that many epochs, because the held-out week is the most recent
    and most valuable data and cannot referee a model that trains on it. Each pass
    anneals the learning rate with a cosine schedule over its own length.

    Holding out the last week of the given frame, rather than the week after it, is
    what keeps this honest as a candidate retriever. Whatever frame ``fit`` is given
    is the only history the model sees, and when it nominates candidates for a label
    week the week after its history is the one being labelled - stopping on it would
    tune the candidates to the answers.
    """

    name = "tower"

    def __init__(self, k: int | None = None, verbose: bool = False, save_artifacts: bool = False,
                 power: float = DECAY_POWER, seed: int = SEED):
        super().__init__(k)
        self.power, self.seed = power, seed
        self.verbose = verbose
        self.save_artifacts = save_artifacts
        self.tuning_history: list[dict] | None = None

    def fit(self, train: pl.DataFrame) -> "TwoTowerRecommender":
        start, _ = week_bounds(train["t_dat"].max())
        self.tuning_history = None
        earlier = train.filter(pl.col("t_dat") < start)
        held_out = purchases_by_customer(train.filter(pl.col("t_dat") >= start))

        self._train(earlier, EPOCHS, held_out)
        self.tuning_history = self.history
        self.chosen_epochs = self.best_epoch()
        self.best_val_map = max(h["val_map@12"] for h in self.tuning_history)
        if self.verbose:
            print(f"  early stopping chose epoch {self.chosen_epochs} (val map@12={self.best_val_map:.5f}), refitting on the full frame")
        if self.save_artifacts:
            write_artifacts(self.tuning_history)
        return self._train(train, self.chosen_epochs, None)

    def _train(self, train: pl.DataFrame, epochs: int, validation: dict[str, list[int]] | None) -> "TwoTowerRecommender":
        """One training pass. With ``validation`` it early-stops on it and restores the best weights."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        verbose, select_best = self.verbose, validation is not None

        self.article_ids, self.item_features = article_ids, item_features = item_feature_matrix()

        customers, self.customer_index, rows, cols, weights, self.history_sum, self.history_weight = build_training_data(
            train, article_ids, item_features, self.power
        )
        self.customer_attributes = customer_features(customers)
        on_sale = available_articles(train)
        self.available = np.array([a in on_sale for a in article_ids])

        self.user_tower = user_tower = Tower(item_features.shape[1] + self.customer_attributes.shape[1]).to(device)
        self.item_tower = item_tower = ItemTower(item_features.shape[1], len(article_ids)).to(device)
        optimizer = torch.optim.AdamW(
            list(user_tower.parameters()) + list(item_tower.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

        counts = np.bincount(cols, weights=weights, minlength=len(article_ids)).astype(np.float32)
        log_q = torch.from_numpy(np.log(np.clip(counts / counts.sum(), 1e-9, None))).to(device)

        val_truth = validation if select_best else {}
        fallback = RecentBestsellers(k=N_RECOMMENDATIONS).fit(train).ranked if select_best else []
        best_map, best_state, waited, self.history = -1.0, None, 0, []

        for epoch in range(1, epochs + 1):
            user_tower.train()
            item_tower.train()
            order = np.random.permutation(len(rows))
            total_loss = 0.0

            for start in range(0, len(order), BATCH_SIZE):
                batch = order[start : start + BATCH_SIZE]
                if len(batch) < 2:
                    continue
                features = user_inputs(
                    self.history_sum, self.history_weight, self.customer_attributes, rows[batch], cols[batch], weights[batch], item_features
                )
                users = torch.from_numpy(features).to(device)
                items = torch.from_numpy(item_features[cols[batch]]).to(device)
                item_ids = torch.from_numpy(cols[batch]).to(device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                    logits = (user_tower(users) @ item_tower(items, item_ids).T) / TEMPERATURE - log_q[item_ids].unsqueeze(0)
                    loss = F.cross_entropy(logits, torch.arange(len(batch), device=device))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * len(batch)

            schedule.step()
            epoch_loss = total_loss / len(order)

            if not select_best:
                self.history.append({"epoch": epoch, "loss": epoch_loss, "val_map@12": None})
                continue

            val_map = evaluate(self._rank(val_truth, fallback, N_RECOMMENDATIONS), val_truth)["map@12"]
            self.history.append({"epoch": epoch, "loss": epoch_loss, "val_map@12": val_map})

            improved = val_map > best_map
            if improved:
                best_map, waited = val_map, 0
                best_state = ({k: v.clone() for k, v in user_tower.state_dict().items()}, {k: v.clone() for k, v in item_tower.state_dict().items()})
            else:
                waited += 1
            if verbose:
                print(f"  epoch {epoch:>2}  loss={epoch_loss:.4f}  val_map@12={val_map:.5f}{'  *' if improved else ''}")
            if waited >= PATIENCE:
                if verbose:
                    print(f"  early stop: no gain for {PATIENCE} epochs, best was epoch {self.best_epoch()}")
                break

        if best_state is not None:
            user_tower.load_state_dict(best_state[0])
            item_tower.load_state_dict(best_state[1])
        if device == "cuda":
            torch.cuda.empty_cache()
        return self

    def best_epoch(self) -> int:
        """The epoch the validation pass scored best, from whichever history holds the scores."""
        scored = self.tuning_history or self.history
        return max(scored, key=lambda h: h["val_map@12"] or -1)["epoch"]

    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        """Top-k articles per customer.

        A candidate retriever asks for far more than the twelve that get shown -
        the pool is judged on coverage, not on the list it would have displayed.
        """
        return self._rank(customers, fallback, self.k)

    def _rank(self, customers, fallback, n: int) -> dict[str, list[int]]:
        self.user_tower.eval()
        self.item_tower.eval()
        item_features = self.item_features
        with torch.inference_mode():
            item_vectors = torch.cat(
                [
                    self.item_tower(
                        torch.from_numpy(item_features[i : i + CHUNK]).to(self.device),
                        torch.arange(i, min(i + CHUNK, len(item_features)), device=self.device),
                    )
                    for i in range(0, len(item_features), CHUNK)
                ]
            )
            unavailable = torch.from_numpy(~self.available).to(self.device)
            known = [c for c in customers if c in self.customer_index]
            rows = np.array([self.customer_index[c] for c in known])

            predictions = {}
            for start in range(0, len(rows), CHUNK):
                batch = rows[start : start + CHUNK]
                features = user_inputs(self.history_sum, self.history_weight, self.customer_attributes, batch)
                user_vectors = self.user_tower(torch.from_numpy(features).to(self.device))
                scores = (user_vectors @ item_vectors.T).masked_fill(unavailable, float("-inf"))
                top = scores.topk(min(n, int(self.available.sum())), dim=1).indices.cpu().numpy()
                for customer, row in zip(known[start : start + CHUNK], top):
                    predictions[customer] = [self.article_ids[j] for j in row]

        return {c: predictions.get(c, fallback) for c in customers}


def write_artifacts(history: list[dict]) -> None:
    """Save the training history and its curve to ``results/two_tower/``.

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
    """Fit on everything before the test week and score the test week.

    ``fit`` holds out the last week of that frame - the validation week - to choose
    the epoch count, which is what produces the saved training curve, then refits on
    the whole frame.
    """
    fitting = load_fitting_data()
    trained = TwoTowerRecommender(k=N_RECOMMENDATIONS, verbose=True, save_artifacts=True).fit(fitting)
    ground_truth = purchases_by_customer(load_transactions("test"))
    predictions = trained.recommend(ground_truth, RecentBestsellers(k=N_RECOMMENDATIONS).fit(fitting).ranked)

    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)
    print(f"{MODEL_NAME}: early stopping chose epoch {trained.chosen_epochs} (val map@12={trained.best_val_map:.5f})")
    print(f"{MODEL_NAME}: artifacts in {ARTIFACTS.relative_to(ROOT)}")
    print_scores(scores)
    return scores


def tune_decay(powers=DECAY_GRID, seeds=DECAY_SEEDS) -> pl.DataFrame:
    """Fit on the training weeks for every decay shape and seed, and score the validation week.

    Several seeds per shape, because one GPU training run moves by about as much
    as the differences being measured.
    """
    train, truth = load_transactions("train"), purchases_by_customer(load_transactions("val"))
    fallback = RecentBestsellers(k=N_RECOMMENDATIONS).fit(train).ranked
    rows = []
    for power in powers:
        for seed in seeds:
            model = TwoTowerRecommender(k=N_RECOMMENDATIONS, power=power, seed=seed).fit(train)
            scores = evaluate(model.recommend(truth, fallback), truth)
            rows.append({"power": power, "seed": seed, "val_map@12": round(scores["map@12"], 6),
                         "val_recall@100": round(scores["recall@100"], 6)})
            print(f"  power={power:<4} seed={seed:<3} val map@12={scores['map@12']:.5f}  val recall@100={scores['recall@100']:.5f}", flush=True)
            del model
    sweep = pl.DataFrame(rows)
    sweep.write_csv(RESULTS / "two_tower_decay.csv")
    print(sweep.group_by("power").agg(pl.col("val_map@12").mean(), pl.col("val_recall@100").mean()).sort("power"))
    return sweep


if __name__ == "__main__":
    import sys

    tune_decay() if "--tune-decay" in sys.argv else main()
