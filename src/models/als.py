"""Model 2 - collaborative filtering with ALS.

The first personalised model. It ignores everything about what an article *is* -
no photos, no descriptions, no categories - and looks only at who bought what.
Purchases go into a big sparse customer x article matrix, and ALS factorises it
into a small dense vector per customer and per article, chosen so that their dot
product reproduces the observed purchases. A customer's recommendations are the
articles whose vectors point the same way as theirs.

What that buys over popularity: the model discovers that people who buy this tend
to also buy that, without anyone describing either item. What it cannot do is say
anything about an article nobody has bought yet - a brand new article has no
column in the matrix, which is the cold-start problem model 3 exists to solve.

Three settings that matter, all chosen on the validation week:

    time decay       a purchase made d days before the end of the history counts
                     alpha / (1 + d) ** p. The two knobs do different jobs. p is
                     the shape - how fast an old purchase fades against a recent
                     one - and alpha only multiplies every weight, so it leaves
                     that ratio alone and sets how loudly a purchase counts against
                     the unobserved cells (which implicit ALS weighs at 1) and
                     where BM25's saturation sits. A 5 x 7 grid on the validation
                     week (``--tune-alpha``) separates them: no decay at all scores
                     0.015, p = 1 tops out at 0.0245, and p = 1.5 at alpha = 80 is
                     best at 0.0256 - last week's basket should count even more
                     than 1 / (1 + d) gave it. p = 2 is a tie with 1.5.
    BM25 weighting   down-weights the bestsellers everyone buys, so the factors
                     spend their capacity on informative co-purchases
                     (val MAP@12 0.0102 -> 0.0141 on undecayed counts).
    no filtering of already-bought items   fashion is heavily repurchased, and
                     suppressing known items cost roughly half the score.

The idea of a time-decayed confidence is borrowed from
JonMcEntee/hm-fashion-recommendations, which weights purchases alpha / (1 + days)
before fitting ALS; the shape exponent and both values were tuned here.

Customers with no training history have no vector at all; they fall back to the
store's bestsellers of the last seven days, which is what a production system
would do. Model 1's four-month list is the wrong fallback for a fashion catalog:
it still carries summer swimwear into late September, and swapping it for the
weekly list more than doubles MAP@12 on the customers who receive it.

Run it with ``python -m src.models.als`` to score it and append the row to
results/metrics_comparison.csv, or with ``--tune-alpha`` to rerun the time-decay
sweep on the validation week, which writes results/als_time_decay.csv and its
chart docs/images/als_time_decay.png.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import polars as pl
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from threadpoolctl import threadpool_limits

from src.data.loading import days_before_end, load_fitting_data, load_transactions, purchases_by_customer
from src.evaluation.metrics import KS, evaluate
from src.evaluation.reporting import print_scores, save_result
from src.models.base import Recommender
from src.models.popularity import RecentBestsellers
from src.paths import RESULTS, ROOT

MODEL_NAME = "02_collaborative_als"
N_RECOMMENDATIONS = max(KS)
FACTORS = 512
REGULARIZATION = 0.05
ITERATIONS = 15
BM25_K1 = 100
BM25_B = 0.8
TIME_DECAY_ALPHA = 80.0
SEED = 42
DECAY_POWER = 1.5
ALPHA_GRID = (5, 10, 20, 40, 80, 160, 320)
POWER_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)


def build_matrix(train: pl.DataFrame, alpha: float = TIME_DECAY_ALPHA, power: float = DECAY_POWER):
    """Sparse customer x article matrix of time-decayed purchase confidence.

    A purchase made ``d`` days before the end of the training window contributes
    ``alpha / (1 + d) ** power``, and repeat purchases of the same article add
    up. Recency is what the decay buys: at the tuned power of 1.5 the same
    purchase is worth about 260x more on the last day of the window than on the
    40th day back.

    Returns the matrix, the article id of each column, and each customer's row.
    """
    users = train["customer_id"].unique().sort().to_list()
    items = train["article_id"].unique().sort().to_list()
    user_index = {user: i for i, user in enumerate(users)}
    item_index = {item: i for i, item in enumerate(items)}

    days_before_split = train.select(days_before_end(train).alias("days"))["days"].to_numpy()
    confidence = (alpha / (1.0 + days_before_split) ** power).astype(np.float32)

    matrix = sp.csr_matrix(
        (
            confidence,
            (train["customer_id"].replace_strict(user_index).to_numpy(), train["article_id"].replace_strict(item_index).to_numpy()),
        ),
        shape=(len(users), len(items)),
    )
    matrix.sum_duplicates()
    return matrix, items, user_index


class AlsRecommender(Recommender):
    """BM25-weighted, time-decayed ALS over the customer x article matrix."""

    name = "als"

    def __init__(self, k: int | None = None, alpha: float = TIME_DECAY_ALPHA, power: float = DECAY_POWER):
        super().__init__(k)
        self.alpha, self.power = alpha, power

    def fit(self, train: pl.DataFrame) -> "AlsRecommender":
        self.matrix, self.items, self.user_index = build_matrix(train, self.alpha, self.power)
        self.weighted = bm25_weight(self.matrix, K1=BM25_K1, B=BM25_B).tocsr()
        self.model = AlternatingLeastSquares(
            factors=FACTORS, regularization=REGULARIZATION, iterations=ITERATIONS, random_state=SEED, num_threads=8
        )
        with threadpool_limits(1, "blas"):
            self.model.fit(self.weighted, show_progress=False)
        return self

    def recommend(self, customers: Iterable[str], fallback: Sequence[int] = ()) -> dict[str, list[int]]:
        """Top-k articles per customer. ``k`` is capped at the catalog size."""
        known = [c for c in customers if c in self.user_index]
        rows = np.array([self.user_index[c] for c in known])
        ranked, _ = self.model.recommend(
            rows, self.weighted[rows], N=min(self.k, len(self.items)), filter_already_liked_items=False
        )
        personalised = {c: [self.items[j] for j in row] for c, row in zip(known, ranked)}
        return {c: personalised.get(c, fallback) for c in customers}


def main() -> dict:
    train, test = load_fitting_data(), load_transactions("test")
    ground_truth = purchases_by_customer(test)

    model = AlsRecommender(k=N_RECOMMENDATIONS).fit(train)
    fallback = RecentBestsellers(k=N_RECOMMENDATIONS).fit(train).ranked

    predictions = model.recommend(ground_truth, fallback)
    covered = sum(1 for c in ground_truth if c in model.user_index)
    assert all(len(p) == N_RECOMMENDATIONS for p in predictions.values())

    scores = evaluate(predictions, ground_truth)
    save_result(MODEL_NAME, scores)

    print(f"{MODEL_NAME}: matrix {model.matrix.shape} nnz={model.matrix.nnz:,}")
    print(f"  personalised for {covered:,} / {len(ground_truth):,} test customers ({100 * covered / len(ground_truth):.0f}%), rest fall back to popularity")
    print_scores(scores)
    return scores


def tune_time_decay(alphas=ALPHA_GRID, powers=POWER_GRID) -> pl.DataFrame:
    """Fit on the training weeks for every (power, alpha) pair and score the validation week.

    The weight ``alpha / (1 + d) ** power`` has two knobs that do different jobs.
    ``power`` is the shape: how fast an old purchase fades relative to a recent
    one. ``alpha`` multiplies every weight by the same constant, so it leaves that
    ratio untouched and only sets the scale - how strongly a purchase counts
    against the unobserved cells, which implicit ALS weighs at 1, and where BM25's
    ``K1`` saturation sits. Sweeping alpha alone, at one shape, says nothing about
    recency; the grid separates the two. The test week is never read.
    """
    train, truth = load_transactions("train"), purchases_by_customer(load_transactions("val"))
    fallback = RecentBestsellers(k=N_RECOMMENDATIONS).fit(train).ranked
    rows = []
    for power in powers:
        for alpha in alphas:
            model = AlsRecommender(k=N_RECOMMENDATIONS, alpha=alpha, power=power).fit(train)
            scores = evaluate(model.recommend(truth, fallback), truth)
            rows.append({"power": power, "alpha": alpha, "val_map@12": round(scores["map@12"], 6),
                         "val_recall@100": round(scores["recall@100"], 6)})
            print(f"  power={power:<4} alpha={alpha:>4}  val map@12={scores['map@12']:.5f}", flush=True)
    sweep = pl.DataFrame(rows)
    sweep.write_csv(RESULTS / "als_time_decay.csv")

    plot_time_decay(sweep)
    return sweep

def plot_time_decay(sweep: pl.DataFrame) -> None:
    """One line per decay shape across the alpha grid, written to docs/images/als_time_decay.png."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    powers = sorted(sweep["power"].unique().to_list())
    alphas = sorted(sweep["alpha"].unique().to_list())
    fig = Figure(figsize=(8.6, 4))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    for colour, power in zip(colours, powers):
        line = sweep.filter(pl.col("power") == power).sort("alpha")
        label = "p = 0  (no decay)" if power == 0 else f"p = {power:g}" + ("  (old setting)" if power == 1 else "")
        ax.plot(line["alpha"], line["val_map@12"], color=colour, linewidth=2, marker="o", markersize=5, label=label)
    best = sweep.sort("val_map@12", descending=True).head(1)
    ax.scatter(best["alpha"], best["val_map@12"], s=120, facecolor="none", edgecolor="#0b0b0b", linewidth=1.5, zorder=3)
    ax.annotate(f"best: p = {best['power'][0]:g}, alpha = {best['alpha'][0]:g}", (best["alpha"][0], best["val_map@12"][0]),
                textcoords="offset points", xytext=(-14, 4), ha="right", va="bottom", fontsize=8, color="#0b0b0b")
    ax.set_xscale("log")
    ax.set_xticks(alphas, [f"{a:g}" for a in alphas])
    ax.minorticks_off()
    ax.set_xlabel("alpha: the scale of every weight", color="#52514e")
    ax.set_ylabel("validation MAP@12", color="#52514e")
    ax.set_title("ALS purchase weight = alpha / (1 + days old)^p", fontsize=10, color="#0b0b0b")
    ax.legend(title="shape p", title_fontsize=8, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ad")
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "images" / "als_time_decay.png", dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    import sys

    tune_time_decay() if "--tune-alpha" in sys.argv else main()
