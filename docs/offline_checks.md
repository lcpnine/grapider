# Offline checks before the next experiment

Date: 2026-10-31. Everything here is exploratory and scored on the tune half only, except the cost count, which uses names and picks across all dev and no labels. No model calls were made, and cache line counts did not change. Gold and the split were not changed.
Code: `prereq/sensitivity.py`. Results: the last four rows of `runs/experiments.jsonl` (`exploratory_*`).

## Co-mention gate

The gate thresholds were written into `prereq/sensitivity.py` before the first run. The handoff gave no numbers, so they are this session's choice. "Shared page" means any of the 31,969 library pages names both concepts.

| Check | Result | Needed | Outcome |
|---|---|---|---|
| Gold-positive pairs with a shared page | 16.4% | ≥ 30% | Fail |
| Uncapped candidate edges with a shared page | 28.0% | ≥ 30% | Fail |
| Lift, gold positives over labeled non-parents | 2.94× [1.95, 4.19] (16.4% vs 5.6%) | ≥ 1.5×, interval above 1 | Pass |
| Direct F1 if a perfect checker removed every labeled-wrong edge with a shared page | .272 → .313 (+15.0%) | ≥ +20% | Fail |

Among labeled candidate edges, a shared page does not separate right from wrong: 26.2% of true edges have one and 27.7% of wrong edges do, a lift of 0.94× [0.57, 1.42]. Direct precision is .218 with a shared page and .232 without.

Decision under the rule: the pair-level evidence arm is not worth running as designed. It would have cost about 4.4M tokens, or 7.5M with the 1.7× overrun seen earlier (1,083 calls per arm, about 2,042 tokens per call).

Limits: broad names inflate coverage ("learning" matches 3,498 pages). Only 49% of lectures were downloadable, and the ceiling trusts gold labels, which may understate precision. This rules out this design on this library, not course materials in general.

## Duplicate concepts

The rule treats two names as one concept if they share the same words apart from generic ones, the same letters, or an abbreviation or initials. It found 10 pairs: 2 inside tune, 6 straddling tune and holdout, 2 outside tune. Straddling pairs merge predictions only, so no holdout label enters tune. Two false matches (`A* search`/`search`, `data structures`/`data structures and algorithms`) were corrected after seeing concept names and before any scoring. Clear paraphrases such as `reading comprehension`/`Machine Reading Comprehension` are not caught.

| Arm | Direct F1, as is → merged | Closure F1, as is → merged |
|---|---|---|
| Capped | .291 → .300 | .289 → .296 |
| Uncapped | .272 → .283 | .275 → .271 |

No conclusion flips:

| Comparison, merged | Direct F1 | Closure F1 |
|---|---|---|
| Uncapped vs capped | −.017 [−.054, .017] | −.025 [−.061, .009] |
| Real excerpts vs uncapped | −.057 [−.099, −.015] | −.060 [−.097, −.023] |
| Real excerpts vs placebo | +.008 [−.045, .062] | −.001 [−.061, .054] |

The repair rows were not rescored, because merging changes edges the checkers never saw.

## Hub sensitivity

**Six of the seven hubs are in the holdout half.** Only `natural language processing intro` is in tune. Tune loses just 26 of its 171 positives when hub positives are removed, against 422 of 951 across all dev, so tune barely tests hub influence. The holdout's labels are mixed very differently from tune's.

| Comparison, F1 | As labeled | Hub positives removed |
|---|---|---|
| Uncapped vs capped, direct | −.019 [−.055, .014] | −.034 [−.072, .002] |
| Uncapped vs capped, closure | −.015 [−.053, .022] | **−.043 [−.080, −.007]** |
| Real excerpts vs uncapped, direct | −.052 [−.092, −.011] | −.037 [−.079, .005] |
| Real excerpts vs uncapped, closure | −.067 [−.106, −.029] | −.041 [−.082, −.003] |
| Real excerpts vs placebo, direct | +.002 [−.056, .055] | −.004 [−.068, .056] |

- **Experiment 1 shifts.** "Closure within noise" becomes "capped is better on closure". Removing every pair with a hub parent gives the same answer (−.042 [−.078, −.006]).
- **Experiment 2 holds.** The real-vs-uncapped direct interval now includes zero, but excerpts still show no gain, so the decision stands.
- **Experiment 3 holds.** Every pass or fail stays the same:

| Repair, hub positives removed | False pairs removed | True pairs lost | Direct recall lost | Closure F1 | Thresholds |
|---|---|---|---|---|---|
| None | – | – | – | .256 | – |
| 50 random edges | 4.0% | 1.7% | 3.9% | .259 | Miss |
| Muse + excerpts says no | 21.2% | 4.0% | 0.8% | .284 | Miss |
| Frozen judge says no | 29.1% | 5.5% | 1.5% | .297 | Miss |
| 50 widest edges, unchecked | 35.3% | 8.7% | 1.5% | .302 | Clears (still a candidate) |
