"""Cross-validation for the final two-stage model.

Every number elsewhere in this project comes from one held-out week scored once.
That is enough to rank five models that differ by large margins, and not enough
to trust a small difference, which is how two rounds of hyperparameter tuning
earlier in this project produced changes that looked real on one week and
vanished on another.

Two different things get confused under the word "variance", so they are
measured apart here:

    across seeds   the same week, re-run with a different random draw for the
                   negative sample and the ranker. This is the noise floor of
                   the pipeline, and any improvement smaller than it is not
                   an improvement.
    across folds   different held-out weeks, each fitted only on what came
                   before it. This is how much the score depends on which week
                   you happened to test on, and it is the larger of the two.

The folds are not interchangeable. Fold 0 holds out the last week and, with the
default expanding window, has the most history to fit on; each earlier fold has
one week less. ``--window-weeks`` holds every fold to the same length of history
instead, and doing so did not shrink the spread across folds: the weeks
genuinely differ, so folds are reported individually rather than only as one
average.

Run it with ``python -m src.evaluation.cross_validation`` for the expanding
window, which writes ``results/cross_validation.csv``, or with
``--window-weeks 16`` for the fixed window, which writes
``results/cross_validation_sliding.csv``. Each writes a ``_summary.csv`` beside it.
"""

from __future__ import annotations

import os

os.environ.setdefault("POLARS_MAX_THREADS", "4")

import argparse
import gc
from pathlib import Path

import polars as pl

from src.paths import RESULTS
from src.ranking.two_stage import evaluate_two_stage

RESULTS_PATH = RESULTS / "cross_validation.csv"
SLIDING_RESULTS_PATH = RESULTS / "cross_validation_sliding.csv"
CV_METRICS = ("map@12", "recall@12", "ndcg@12", "hitrate@12", "recall@100")


class RollingOriginValidator:
    """Re-runs the whole two-stage pipeline over several weeks and seeds.

    Nothing is reused between runs: each fold refits the retrievers, rebuilds
    the label weeks and retrains the ranker on only the data before its own
    evaluation week. That is slow, and it is the point, since sharing any fitted
    object across folds would leak the week being scored.
    """

    def __init__(self, folds: int = 3, seeds: tuple[int, ...] = (42,), verbose: bool = True,
                 window_weeks: int | None = None):
        self.folds = folds
        self.seeds = tuple(seeds)
        self.verbose = verbose
        self.window_weeks = window_weeks
        self.runs: pl.DataFrame | None = None

    def run(self) -> pl.DataFrame:
        """One row per (fold, seed), holding both stages' scores."""
        rows = []

        for fold in range(self.folds):
            for seed in self.seeds:
                result = evaluate_two_stage(verbose=False, save=False, weeks_back=fold, seed=seed,
                                            window_weeks=self.window_weeks)
                scores, retrieval = result.scores, result.retrieval_scores
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "window_weeks": self.window_weeks or 0,
                        "n_customers": scores["n_customers"],
                        "ceiling": round(result.ceiling, 5),
                        **{metric: round(scores[metric], 6) for metric in CV_METRICS},
                        "retrieval_map@12": round(retrieval["map@12"], 6),
                    }
                )
                if self.verbose:
                    print(f"  fold {fold} seed {seed}: map@12={scores['map@12']:.5f} "
                          f"ceiling={result.ceiling:.4f} n={scores['n_customers']:,}", flush=True)
                del result, scores, retrieval
                gc.collect()

        self.runs = pl.DataFrame(rows)
        return self.runs

    def summary(self) -> pl.DataFrame:
        """Mean of each metric with the two spreads that matter beside it."""
        if self.runs is None:
            raise RuntimeError("call run() first")

        rows = []
        for metric in (*CV_METRICS, "retrieval_map@12"):
            per_fold = self.runs.group_by("fold").agg(pl.col(metric).mean().alias("mean"))["mean"]
            seed_spread = self.runs.group_by("fold").agg(pl.col(metric).std().alias("std"))["std"].mean()
            rows.append(
                {
                    "metric": metric,
                    "mean": round(self.runs[metric].mean(), 6),
                    "std_across_folds": round(per_fold.std(), 6) if self.folds > 1 else None,
                    "std_across_seeds": round(seed_spread, 6) if seed_spread is not None else None,
                    "min": round(self.runs[metric].min(), 6),
                    "max": round(self.runs[metric].max(), 6),
                }
            )
        return pl.DataFrame(rows)

    def save(self, path: Path = RESULTS_PATH) -> Path:
        if self.runs is None:
            raise RuntimeError("call run() first")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.runs.write_csv(path)
        self.summary().write_csv(path.with_name(f"{path.stem}_summary.csv"))
        return path


def main() -> pl.DataFrame:
    parser = argparse.ArgumentParser(description="Cross-validate the two-stage model over four weeks and three seeds.")
    parser.add_argument("--window-weeks", type=int, default=None,
                        help="fix every fold to this many weeks of history instead of an expanding window")
    args = parser.parse_args()

    validator = RollingOriginValidator(folds=4, seeds=(42, 7, 13), window_weeks=args.window_weeks)
    validator.run()
    print(validator.summary())
    path = RESULTS_PATH if args.window_weeks is None else SLIDING_RESULTS_PATH
    print(f"written to {validator.save(path)}")
    return validator.runs


if __name__ == "__main__":
    main()
