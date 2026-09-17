"""Writing scores to disk and to the terminal.

``src/evaluation/metrics.py`` computes numbers and never touches a file. This
module is where they leave the process: one upserted row per model in
``results/metrics_comparison.csv``, the two-stage model's full report, and the
per-k lines every model prints when run directly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.metrics import KS, METRICS
from src.paths import RESULTS

RESULTS_PATH = RESULTS / "metrics_comparison.csv"
REPORT_PATH = RESULTS / "two_stage_report.json"


def round_floats(values: dict, digits: int = 6) -> dict:
    """Round the float values of a flat dict, leaving counts and names untouched."""
    return {k: round(v, digits) if isinstance(v, float) else v for k, v in values.items()}


def print_scores(scores: dict) -> None:
    for k in KS:
        print(f"  @{k:<4} " + "  ".join(f"{m}={scores[f'{m}@{k}']:.5f}" for m in METRICS))


def save_result(model_name: str, scores: dict, path: Path = RESULTS_PATH) -> Path:
    """Upsert one model's row into the shared comparison file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {"model": model_name, **round_floats(scores)}
    rows = []
    if path.exists():
        with path.open() as f:
            rows = [r for r in csv.DictReader(f) if r["model"] != model_name]
    rows.append(row)
    rows.sort(key=lambda r: r["model"])

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_report(scores, retrieval_scores, segments, catalog_shape, ceiling, pool_rows,
                 path: Path = REPORT_PATH) -> Path:
    """Everything about the final model that the README quotes, as one file.

    The README used to be written by hand from whatever run happened to be on
    screen, which is how it twice ended up disagreeing with the results file.
    Anything the README states about this model is regenerated from here.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {
            "ceiling": round(ceiling, 5),
            "pool_rows": pool_rows,
            "candidates_per_customer": round(pool_rows / scores["n_customers"]),
            "final": round_floats(scores),
            "retrieval": round_floats(retrieval_scores),
            "segments": {name: round_floats(part) for name, part in segments.items()},
            "catalog_shape": round_floats(catalog_shape, 5),
        },
        indent=2,
    ))
    return path
