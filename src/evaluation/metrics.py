"""Shared ranking metrics for every Aurora model.

Every model emits predictions in the same shape, so this module is imported
unmodified by all five models:

    predictions   {customer_id: [article_id, ...]}  ranked best first
    ground_truth  {customer_id: [article_id, ...]}  test-week purchases

Metrics are averaged over customers that actually bought something in the test
week; customers a model returns nothing for score zero rather than being
dropped. MAP@k and NDCG@k use the Kaggle H&M convention of dividing by
min(n_relevant, k), so a customer who bought fewer than k items can still
reach 1.0.
"""

from __future__ import annotations

import math

KS = (12, 50, 100)
METRICS = ("precision", "recall", "hitrate", "ndcg", "map")


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


def evaluate_segments(predictions, ground_truth, segments, ks=KS) -> dict:
    """Score one set of predictions separately for each named group of customers.

    ``segments`` maps a name to the customer ids belonging to it. An aggregate
    score hides where it came from: a model can look strong overall and owe all
    of it to customers who already had long histories, which is exactly the
    group a recommender needs least help with.
    """
    scored = {}
    for name, members in segments.items():
        truth = {c: items for c, items in ground_truth.items() if c in members}
        if truth:
            scored[name] = evaluate(predictions, truth, ks=ks)
    return scored


def beyond_accuracy(predictions, popularity, category_of, catalog_size, k=12) -> dict:
    """What the top-k lists look like as a whole, rather than how often they hit.

    Accuracy alone cannot tell apart a recommender that understands customers
    from one that has learned to show everybody the same few hundred bestsellers,
    because on a skewed catalog that strategy scores respectably.

        coverage   share of the catalog appearing in at least one top-k list
        novelty    mean self-information of the recommended articles, the
                   negative log2 of each one's share of all purchases, so
                   obvious choices score low
        diversity  distinct product types within a top-k list over k, averaged
                   over customers

    ``popularity`` is ``{article_id: purchase count}`` over the fitting data and
    ``category_of`` is ``{article_id: product type}``.
    """
    total = sum(popularity.values())
    seen, novelty, diversity, n = set(), 0.0, 0.0, 0

    for items in predictions.values():
        top = _dedup(items)[:k]
        if not top:
            continue
        n += 1
        seen.update(top)
        novelty += sum(-math.log2((popularity.get(a, 0) + 1) / (total + catalog_size)) for a in top) / len(top)
        diversity += len({category_of.get(a) for a in top}) / len(top)

    if not n:
        raise ValueError("predictions contain no ranked lists")
    return {
        f"coverage@{k}": len(seen) / catalog_size,
        f"novelty@{k}": novelty / n,
        f"diversity@{k}": diversity / n,
    }


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

    parts = evaluate_segments(preds, truth, {"hit": {"u1"}, "miss": {"u2"}}, ks=(3,))
    assert abs(parts["hit"]["map@3"] - 0.583333) < 1e-6, parts
    assert parts["miss"]["map@3"] == 0.0 and parts["hit"]["n_customers"] == 1

    shape = beyond_accuracy({"u1": ["a", "b"], "u2": ["a", "b"]}, {"a": 30, "b": 10}, {"a": 1, "b": 1}, 4, k=2)
    assert abs(shape["coverage@2"] - 0.5) < 1e-9, shape
    assert abs(shape["diversity@2"] - 0.5) < 1e-9, shape
    assert abs(shape["novelty@2"] - (-math.log2(31 / 44) - math.log2(11 / 44)) / 2) < 1e-9, shape

    rare = beyond_accuracy({"u": ["c"]}, {"a": 30, "b": 10}, {"c": 9}, 4, k=1)
    assert rare["novelty@1"] > shape["novelty@2"] and rare["diversity@1"] == 1.0
    print("metrics self-check passed")


if __name__ == "__main__":
    _self_check()
