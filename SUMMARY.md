# Aurora — full project summary

A working note, not documentation. It records what exists, what every number is,
what was learned, and which parts are probably not worth keeping in the README.
Written so the keep-or-cut decisions can be made from one page.

---

## 1. What the project is

A fashion recommender built on the Kaggle H&M dataset, structured as a clean
progression from a naive baseline to a two-stage retrieval-and-ranking system.
Every model emits the same prediction format and is scored by the same metrics
module, so the final comparison table is like for like.

Hardware is a single RTX 4060 with 8 GB of VRAM, which is the reason for the
sampling, the frozen image encoder, and the batching in the ranker.

## 2. The data

Transactions are filtered to the **last 140 days first**, then 6% of the
customers active in that window are sampled, then only their in-window
transactions are kept. Doing it in this order is what stops the sample being
dominated by an old catalog.

| | value |
|---|---|
| transactions | 382,423 |
| customers | 39,540 |
| articles | 28,086 |
| window | 2020-05-06 to 2020-09-22 |
| train | 352,678 rows, to 2020-09-08 |
| val | 15,184 rows, 09-09 to 09-15 |
| test | 14,561 rows, 09-16 to 09-22 |
| test customers scored | 4,155 |

Split is by time, never by user. Models tune on the validation week and then
refit on train plus val before predicting test, which is what a weekly
production retrain does. That single change roughly doubled four of five models
earlier in the project and remains the largest win in it.

## 3. The five models

| # | model | what it adds |
|---|---|---|
| 1 | Popularity | non-personalised floor |
| 2 | ALS | co-purchase structure |
| 3 | Content-based | metadata + CLIP image vectors, reaches cold items |
| 4 | Two-tower | learned user and item towers, in-batch negatives, logQ correction |
| 5 | Two-stage | pools candidates from two retrievers, reranks with LightGBM LambdaRank |

Model 5 is the final system: `RecentBestsellers(k=700)` plus
`TwoTowerRetriever(k=700)` produce about 1,055 candidates per customer, and a
LambdaRank model with 17 features orders them. It trains on four rolling-origin
label weeks with 60 negatives per customer and 600 trees.

## 4. Current results

Test week, 4,155 customers, all five models identical evaluation.

| model | MAP@12 | Hit@12 | NDCG@12 | R@100 |
|---|---|---|---|---|
| 01 popularity | 0.00350 | 0.03201 | 0.00715 | 0.04052 |
| 02 ALS | 0.02434 | 0.09338 | 0.03547 | 0.09462 |
| 03 content | 0.01833 | 0.06065 | 0.02492 | 0.07272 |
| 04 two-tower | 0.01970 | 0.08688 | 0.03013 | 0.13525 |
| 05 two-stage | **0.03121** | **0.13141** | **0.04694** | **0.17718** |

**The 0.03121 is not stable.** Eight recorded runs of this exact configuration on
this exact week span 0.02925 to 0.03121, mean 0.02980, standard deviation
0.00062. The cause is GPU non-determinism in the two-tower, which is retrained
once per label week and feeds the candidate pool. Models 1, 2 and 3 are fully
deterministic and reproduce byte-identically; models 4 and 5 do not.

Consequence: the run currently in the table is the **highest of the eight**.
Quoting it gives "+28% over ALS"; the mean gives **+22%**, which is the honest
figure.

## 5. What the last two rounds added

### 5.1 A real bug, fixed

`AlsRetriever` and `ContentRetriever` asked their underlying model for 100
candidates and then sliced to `k`. Any `k` above 100 silently returned 100. Not
active, since the default pool no longer uses them, but it made every earlier
experiment that repooled them meaningless. Both now pass `k` through. Verified:
k of 100 / 300 / 700 returns 100 / 300 / 700.

### 5.2 Stage-by-stage scoring

The pool arrives already ordered by `best_rank`, the best position any retriever
gave a candidate. Reading it out in that order scores retrieval alone.

| stage | MAP@12 | R@12 | NDCG@12 | Hit@12 | R@100 |
|---|---|---|---|---|---|
| 1. retrieval, pool order | 0.01616 | 0.04433 | 0.02716 | 0.09170 | 0.15234 |
| 2. reranked | 0.03121 | 0.06950 | 0.04694 | 0.13141 | 0.17718 |

The ranker is worth **+93%** on MAP@12 over the pool's own ordering. The pool
makes **47.5%** of test purchases reachable and the ranker converts **37%** of
that into recall@100.

This is the most interview-relevant addition. "Why two stages" now has a number.

### 5.3 Customer segments

Split by how much history the model had on them.

| segment | customers | MAP@12 | R@100 |
|---|---|---|---|
| cold, no history | 848 | 0.00955 | 0.10617 |
| light, 1-4 | 725 | 0.03785 | 0.17805 |
| heavy, 5+ | 2,582 | 0.03646 | 0.20027 |

Two findings. Cold customers score about a quarter of the others, and they are
20% of the test set, so a fifth of the evaluation is effectively measuring the
bestseller list. And light buyers slightly beat heavy ones on MAP@12 while losing
on recall@100, because MAP divides by how much the customer bought: a heavy buyer
has more to find in the same twelve slots.

These numbers move a few points between runs, same non-determinism as above.

### 5.4 Beyond-accuracy metrics

| measure | value | meaning |
|---|---|---|
| coverage@12 | 0.0407 | share of the 28,086 articles appearing in anyone's top 12 |
| novelty@12 | 12.88 bits | how un-obvious the recommended articles are |
| diversity@12 | 0.405 | distinct product types within one list of 12, over 12 |

Coverage of 4% is the interesting one: the system recommends about 1,140 distinct
articles across 4,155 customers. Accuracy metrics cannot see this.

### 5.5 Cross-validation

`src/validation.py`, `RollingOriginValidator`. Four held-out weeks by three seeds,
twelve full pipeline runs, nothing shared between folds.

| fold | evaluates | mean MAP@12 |
|---|---|---|
| 0 | 09-16 to 09-22 (the reported test week) | 0.02966 |
| 1 | 09-09 to 09-15 | 0.02866 |
| 2 | 09-02 to 09-08 | 0.02309 |
| 3 | 08-26 to 09-01 | 0.02337 |

| spread | MAP@12 |
|---|---|
| across seeds, same week | **0.00036** |
| across folds, different weeks | **0.00345** |
| four-week mean | **0.02620** |

**The headline finding of the whole round:** which week you evaluate on moves the
score about ten times more than the random seed, and both are larger than every
hyperparameter change made in this project. The reported test week is the most
favourable of the four.

### 5.6 Sliding-window experiment

The folds above differ in two ways at once: which week they predict, and how much
history they fit on, which falls 15% from fold 0 to fold 3. Repeating everything
with a fixed 16-week window makes training size vary by 3% instead, isolating the
week. Fold 3 is the control, since it already had 16 weeks and loses nothing.

| fold | weeks cut | expanding | fixed 16w | change |
|---|---|---|---|---|
| 0 | 3 | 0.02966 | 0.03093 | +4.3% |
| 1 | 2 | 0.02866 | 0.02913 | +1.7% |
| 2 | 1 | 0.02309 | 0.02358 | +2.1% |
| 3 | 0 | 0.02337 | 0.02338 | **+0.1%** |
| | | std 0.00345 | std 0.00385 | |

Two separate results.

**The weeks genuinely differ.** The spread did not collapse when data was
equalised; it widened slightly. Data volume was never the explanation.

**Older data is mildly harmful.** The gain tracks how much history was removed,
and the control lands on +0.1%, which is the check that the window code is
correct. On a catalog that turns over weekly, May purchases describe an
assortment that no longer exists.

**Not adopted.** Using it honestly means re-running all five models on a shorter
window and tuning the window length itself. It is recorded, not applied.

### 5.7 Hypotheses tested and rejected

Worth keeping because they are what makes the conclusion credible.

- **Cold-customer mix explains the fold spread.** Rejected. Cold share moves only
  20.4% to 22.8% across folds, worth about 0.0003 of a 0.0066 gap.
- **Shrinking training history explains it.** Rejected by the sliding window.
- **Repeat-purchase rate explains it.** Partially. It drops from 2.96% to 2.45%
  in the weaker weeks and is the highest-precision signal the ranker has, but it
  is not large enough to carry the gap. Direction right, magnitude wrong.

### 5.8 Infrastructure

- `run()` writes `results/two_stage_report.json` with every number the README
  quotes about the final model.
- The README has **8 generated blocks** rewritten from `results/` by a script,
  with assertions against doubled pipes and ragged rows. The README had twice
  drifted from the results file before this.
- Fixed `RollingOriginValidator.save`, which hardcoded the summary filename and
  so overwrote the expanding summary with the sliding one.

### 5.9 Documentation drift found and corrected

All of these were in the README or docstrings, all wrong against the code:

- a six-retriever pool that is actually two
- a 0.32 recall ceiling that is actually 0.475
- "thirteen features" that are seventeen
- "thirty negatives" that are sixty
- "0.17% positives" that is 0.13%
- a CLIP paragraph duplicated verbatim

## 6. Open issues

1. **The reported model 5 row is whichever run last executed**, so it moves on
   every notebook rebuild. Currently sitting on the luckiest of eight. Fix is to
   pin it to the mean of recorded runs. **Not done, awaiting a decision.**
2. **The API artifacts are from a different run** (0.02984) than the results file
   (0.03121). Each is internally consistent, but they do not match each other.
3. **Two-tower variance is larger than the two-stage variance** and propagates
   into it. Never characterised on its own.
4. **No tests.** Declined deliberately. `src/metrics.py` has a `_self_check`.
5. **Sliding window not adopted** despite being worth about +4%.

## 7. What I would cut from the README

The README is **719 lines**. The measurement material added in the last two
rounds is **about 210 of them**, and it is the densest, least readable part.

| section | lines | verdict |
|---|---|---|
| `#### What each stage is worth` | 16 | **keep** — answers "why two stages", cheapest high value in the file |
| `#### Who the model actually helps` | 25 | **keep, trim to ~12** — the cold/light/heavy table earns its place, the explanation around it does not need three paragraphs |
| `#### What the recommendations look like` | 41 | **cut to ~8** — keep the coverage number in one sentence, drop novelty and diversity. Novelty in bits is not something a reader can calibrate |
| `## How solid are these numbers` | **127** | **cut to ~30** — this is the bloat. See below |
| `## What these numbers actually mean` | 22 | keep, it is good context |

**Specifically for the 127-line validation section**, the three things genuinely
worth stating are:

1. Run-to-run noise is 0.0006, so differences under about 0.0012 are meaningless.
2. Across four weeks the model scores 0.0262 on average and the reported week is
   the best of them.
3. The week matters ten times more than the seed, which is why tuning stopped.

Everything else — the fold construction table, the zero-gap property, the
confound discussion, the sliding-window table, the rejected hypotheses — is
genuinely interesting and belongs in the **notebook**, where a reader who wants
it can find it, rather than in a README someone skims in two minutes.

That would take the README from 719 lines to roughly **560**, with nothing lost
that a recruiter or interviewer would miss.

## 8. Layout

```
data/          sampling, image embedding extraction
src/           metrics.py, data_utils.py, retrievers.py, validation.py
models/        01..05, one file per model
results/       metrics_comparison.csv, two_stage_report.json,
               cross_validation*.csv
notebooks/     end_to_end.ipynb — 76 cells, 22 figures
api/ web/      FastAPI service and React front end
docs/images/   13 figures, all exported from the notebook
```

| file | lines |
|---|---|
| `models/05_two_stage_ranker.py` | 568 |
| `models/04_two_tower.py` | 417 |
| `src/retrievers.py` | 241 |
| `src/metrics.py` | 187 |
| `src/validation.py` | 130 |
| `src/data_utils.py` | 83 |

## 9. Commits in these rounds

```
36de0e0  Settle the fold spread with a fixed-length training window
44ec7d1  Measure what the single test week was hiding
f5895d3  Give the ranker seventeen features; find that its own settings barely matter
a442121  Rebuild the two-tower and repool retrieval around it
```

## 10. For the resume

Quote **+22% MAP@12** and **+85% Recall@100** over ALS. Both are computed against
the mean of recorded runs, not the single lucky one currently in the table. The
run sitting in the table right now would say +28% and +87%; those are the top of
the range, not the middle.
Metrics to name: MAP@12, Recall@100, NDCG@12, hit rate. Do not quote absolute
values.
