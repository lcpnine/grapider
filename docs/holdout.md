# Holdout result

Date: 2026-11-01. Scored once from `data/splits/exp_freeze.json` (sha256 `f6f0ca0aaadb`), shown to the user first. A second run is refused. It made no model calls: cache line counts did not change. The result is the `stage: holdout` row in `runs/experiments.jsonl` (code hash `80a3941e178b`, on commit `0ab3d70` with the holdout code uncommitted). A dry run on tune first reproduced the logged tune results.

Scope: 159 holdout concepts, 296 positive pairs, 3 runs per arm. Intervals are paired 95% from resampling targets.

## Pre-registered decision

**Widest-50 is rejected.** It missed the true-pairs threshold, so it stays out of the pipeline.

| Threshold, mean of 3 runs | Needed | Holdout | Each run | Outcome |
|---|---|---|---|---|
| False closure pairs removed | ≥ 30% | 43.6% | 40.3, 40.1, 50.4 | Pass |
| True closure pairs lost | < 10% | 15.6% | 19.9, 9.8, 17.2 | **Fail** |
| Direct recall lost | < 5% | 4.0% | 3.9, 3.9, 4.3 | Pass |

Control, 50 random edges (decides nothing): 4.9% false removed, 2.3% true lost, 3.9% direct recall lost, closure F1 .343. On tune, widest-50 lost 8.9% of true pairs, so the tune pass did not carry over.

## Direct edges

| Arm | Edges (all dev) | Precision | Recall | F1 | F1 by run |
|---|---|---|---|---|---|
| Capped | 713 | .325 | .152 | .207 | .216, .202, .203 |
| Uncapped | 1,305 | .316 | .256 | .283 | .285, .290, .273 |
| Uncapped minus widest 50 | 1,255 | .313 | .246 | .275 | .279, .280, .266 |

## Closure

| Arm | Precision | Recall | F1 | F1 by run |
|---|---|---|---|---|
| Capped | .348 | .362 | .355 | .352, .357, .356 |
| Uncapped | .275 | .449 | .341 | .365, .351, .310 |
| Uncapped minus widest 50 | .363 | .379 | .371 | .370, .393, .349 |

## Paired gains in F1

| Comparison | Direct | Closure |
|---|---|---|
| Uncapped vs capped | +.076 [.040, .111] | −.014 [−.034, .007] |
| Widest-50 vs uncapped | −.007 [−.015, −.001] | +.030 [.007, .052] |
| Widest-50 vs capped | +.068 [.031, .104] | +.016 [−.008, .041] |

Uncapped vs capped is report-only, by the freeze. The cap was removed by design direction, and this result does not restore it. On closure, uncapped trades precision for recall: precision −.073 [−.093, −.051], recall +.087 [.056, .120].

## Sensitivity: hub positives removed (decides nothing)

129 of the 296 positives remain.

| Arm | Direct P / R / F1 | Closure P / R / F1 |
|---|---|---|
| Capped | .261 / .256 / .258 | .170 / .324 / .223 |
| Uncapped | .228 / .375 / .283 | .132 / .412 / .200 |
| Uncapped minus widest 50 | .222 / .351 / .272 | .186 / .352 / .243 |

| Comparison | Direct | Closure |
|---|---|---|
| Uncapped vs capped | +.025 [−.023, .073] | −.023 [−.041, −.003] |
| Widest-50 vs uncapped | −.012 [−.024, −.002] | +.043 [.025, .060] |
| Widest-50 vs capped | +.014 [−.036, .063] | +.020 [−.005, .044] |

Widest-50 also fails here on true pairs lost (14.7%) and direct recall lost (6.4%). It removes 42.7% of false pairs.

## Exploratory readings (not pre-registered)

- **Tune and holdout disagree on direct uncapped vs capped:** −.019 [−.055, .014] on tune, +.076 [.040, .111] here. Without hub positives the holdout gain is +.025 [−.023, .073], so the gain leans on those labels. The hub parents are mostly in this half.
- **Closure scores lean on hub labels:** uncapped closure precision is .275 as labeled and .132 without hub positives. Absolute closure precision here should not be read as graph quality until the labels have human review.
- **Direct precision stays low:** about .32 as labeled and .23 without hub positives in every arm. Nothing scored here improves it, and the main direct-edge problem remains unsolved.

## Limits

- The holdout is not fully unseen. The earlier all-dev diagnosis touched its concepts. The widest-edge wrong share and the gold-limit repair row used all-dev labels and were seen before widest-50 became a candidate. The tune hub check revealed that 6 of 7 hubs are in this half. Any bias from these would most likely favor widest-50, which still failed.
- On average, 40 of the 50 removed edges touch the holdout and 9 lie inside it, so its direct effect rests on few edges. The budget of 50 is fixed and does not scale with graph size.
- Gold is the merged either-dataset label. Independent human review is still pending.
- The pair-level evidence arm was never run, and source and placebo were not scored here.
- Slide text was not sent anywhere. This scoring used saved picks only.
