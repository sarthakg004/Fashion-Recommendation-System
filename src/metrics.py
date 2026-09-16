"""Shared ranking metrics for every Aurora model.

Every model emits predictions in the same shape, so this module is imported
unmodified by all five model scripts:

    predictions   {customer_id: [article_id, ...]}  ranked best first
    ground_truth  {customer_id: [article_id, ...]}  test-week purchases

Metrics are averaged over customers that actually bought something in the test
week; customers a model returns nothing for score zero rather than being
dropped. MAP@k and NDCG@k use the Kaggle H&M convention of dividing by
min(n_relevant, k), so a customer who bought fewer than k items can still
reach 1.0.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

KS = (12, 50, 100)
METRICS = ("precision", "recall", "hitrate", "ndcg", "map")
RESULTS_PATH = Path(__file__).resolve().parents[1] / "results" / "metrics_comparison.csv"


def _dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _dedup(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def evaluate(predictions, ground_truth, ks=KS) -> dict:
    """Score ranked predictions against held-out purchases.

    Returns a flat dict of ``{metric}@{k}`` floats plus ``n_customers``.
    """
    totals = {f"{m}@{k}": 0.0 for k in ks for m in METRICS}
    n_customers = 0

    for customer, truth in ground_truth.items():
        truth = set(truth)
        if not truth:
            continue
        n_customers += 1
        ranked = _dedup(predictions.get(customer, []))

        for k in ks:
            hits = [1.0 if a in truth else 0.0 for a in ranked[:k]]
            n_hits = sum(hits)
            ideal = min(len(truth), k)

            totals[f"precision@{k}"] += n_hits / k
            totals[f"recall@{k}"] += n_hits / len(truth)
            totals[f"hitrate@{k}"] += 1.0 if n_hits else 0.0
            totals[f"ndcg@{k}"] += _dcg(hits) / _dcg([1.0] * ideal)

            running = average_precision = 0.0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    running += 1.0
                    average_precision += running / rank
            totals[f"map@{k}"] += average_precision / ideal

    if not n_customers:
        raise ValueError("ground_truth contains no customers with purchases")

    scores = {name: total / n_customers for name, total in totals.items()}
    scores["n_customers"] = n_customers
    return scores


def save_result(model_name: str, scores: dict, path: Path = RESULTS_PATH) -> Path:
    """Upsert one model's row into the shared comparison file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = {"model": model_name, **{k: round(v, 6) if isinstance(v, float) else v for k, v in scores.items()}}
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


def _self_check() -> None:
    truth = {"u1": ["a", "b"], "u2": ["c"]}
    preds = {"u1": ["x", "a", "b"]}
    s = evaluate(preds, truth, ks=(3,))

    assert abs(s["precision@3"] - (2 / 3) / 2) < 1e-9, s
    assert abs(s["recall@3"] - 1.0 / 2) < 1e-9, s
    assert abs(s["hitrate@3"] - 0.5) < 1e-9, s
    assert abs(s["ndcg@3"] - 0.693426 / 2) < 1e-6, s
    assert abs(s["map@3"] - 0.583333 / 2) < 1e-6, s
    assert s["n_customers"] == 2

    perfect = evaluate({"u": ["a", "b"]}, {"u": ["a", "b"]}, ks=(12,))
    assert abs(perfect["map@12"] - 1.0) < 1e-9 and abs(perfect["ndcg@12"] - 1.0) < 1e-9
    assert abs(perfect["recall@12"] - 1.0) < 1e-9 and abs(perfect["precision@12"] - 2 / 12) < 1e-9

    assert evaluate({"u": ["a", "a", "b"]}, {"u": ["a"]}, ks=(2,))["precision@2"] == 0.5
    print("metrics self-check passed")


if __name__ == "__main__":
    _self_check()
