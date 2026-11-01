# Experiments 1–3: cap removal, source-grounded inference, graph repair

Date: 2026-10-31. Data: LectureBank LB1 and LB2 dev concepts (318). Model: `muse-spark-1.3`. Frozen judge: `gpt-6-luna` with `v1_direct`, names only.
Raw results: `runs/experiments.jsonl`. Code: `prereq/experiment.py`, `prereq/materials.py`.

All numbers are from the tune half (159 concepts, 171 positive pairs), averaged over 3 runs, with paired 95% intervals from resampling targets. Direct scores raw model picks against merged labels. Closure scores the acyclic graph against gold reachability. The holdout half and the final test were not scored.

## Outcome

| Change | Result | Decision |
|---|---|---|
| Remove the 3-parent cap | Recall up, precision down, F1 within noise | Removed from the pipeline; see `docs/cap_removal.md` |
| Source-grounded inference (target-only excerpts) | Worse than uncapped, no better than placebo | Not kept |
| Repair by checker (frozen judge, Muse + excerpts) | Neither meets the threshold | Not kept |
| Remove the 50 widest edges, unchecked | Cleared the thresholds on tune, found after seeing results; lost 15.6% of true pairs on the holdout | Rejected; see `docs/holdout.md` |

The holdout was deliberately not scored during these experiments. Nothing that passed needed confirming, the widest-edge rule was found after looking at results, and the holdout can be scored once. It was scored later, with the result in `docs/holdout.md`.

## Experiment 1: remove the cap

| | Direct P | Direct R | Direct F1 | Closure P | Closure R | Closure F1 | Raw closure F1 | Edges |
|---|---|---|---|---|---|---|---|---|
| Capped at 3 | .339 | .255 | .291 | .262 | .323 | .289 | .229 | 713 |
| Uncapped | .244 | .308 | .272 | .193 | .479 | .275 | .167 | 1,305 |

Uncapped minus capped:

| | F1 | Precision | Recall |
|---|---|---|---|
| Direct | −.019 [−.055, .014] | −.094 [−.143, −.048] | +.053 [.023, .085] |
| Closure | −.015 [−.053, .022] | −.070 [−.107, −.034] | +.156 [.108, .204] |
| Raw closure | −.062 [−.091, −.033] | −.059 [−.087, −.036] | +.113 [.063, .165] |

Repeated uncapped runs share about 68% of edges (Jaccard .676–.686). Vote count is a weak confidence signal: direct precision is .211, .106 and .291 for edges chosen by 1, 2 and 3 runs. The majority graph has direct F1 .260 and closure F1 .288.

## Experiment 2: course materials

Materials gate, passed: 833 of 1,717 lectures downloaded, 259 of 318 concepts covered (81%), gold parents named in their target's excerpts 5.92% vs 1.24% for other concepts (lift 4.76x).

Scores on targets with excerpts only:

| | Direct P | Direct R | Direct F1 | Closure P | Closure R | Closure F1 |
|---|---|---|---|---|---|---|
| Capped | .302 | .222 | .255 | .255 | .304 | .277 |
| Uncapped | .234 | .276 | .253 | .201 | .458 | .279 |
| Real excerpts | .232 | .178 | .201 | .174 | .270 | .212 |
| Placebo excerpts | .360 | .138 | .200 | .375 | .161 | .225 |

| Comparison | Direct F1 | Closure F1 |
|---|---|---|
| Real vs uncapped | −.052 [−.092, −.011] | −.067 [−.106, −.029] |
| Real vs placebo | +.002 [−.056, .055] | −.014 [−.073, .046] |
| Placebo vs uncapped | −.054 [−.112, .002] | −.054 [−.107, .001] |

The keep rule needed a gain of at least 20% over both uncapped and placebo, so source inference fails. The result fits no row of the pre-registered table exactly (real ≈ placebo < uncapped). The nearest is "framing, not content".

Follow-up analysis, not pre-registered:

- Only 9 of 152 gold parents are named in their target's excerpts.
- The model reads the excerpts: recall on those 9 rises from .59 to .85. Recall on the other 143 falls from .26 to .14.
- Share of picks named in the excerpts rises from 15% to 28%, but named picks are not more precise (.21 vs .24 for unnamed).
- Edges real excerpts add have precision .11 (36 scored). Uncapped edges they drop have .19, and edges they keep have .29.
- Placebo keeps uncapped edges with precision .38 and drops edges with .18, so excerpts of any kind raise the model's bar.

This rules out target-only retrieval with this one prompt, not course materials in general.

## Experiment 3: graph repair

Applied to the uncapped graph. Thresholds: at least 30% of false closure pairs removed, under 10% of true pairs lost, under 5% of direct recall lost. "Widest" means the edges that alone support the most closure pairs, a count that uses no labels. Checks covered the 50 widest edges.

| Repair | Edges removed | False pairs removed | True pairs lost | Direct recall lost | Closure F1 |
|---|---|---|---|---|---|
| None | 0 | – | – | – | .274 |
| 50 random edges | 50 | 4.0% | 1.9% | 3.9% | .276 |
| Muse + excerpts says no | 24 | 21.3% | 3.9% | 0.7% | .303 |
| Frozen judge says no | 30 | 29.1% | 5.4% | 1.3% | .315 |
| 50 widest edges, unchecked | 50 | 35.4% | 8.9% | 1.3% | .319 |
| Gold-label upper limit, 50 edges | 50 | 41.6% | 8.9% | 0% | .336 |

- Frozen judge: misses the threshold (23%, 32%, 32% across runs). Against random picks of the same size from the same 50 edges, only 4.6% of draws did as well on closure F1, but the mean gain is just .011.
- Muse + excerpts: fails, and 35% of the same random draws did as well.
- Reverse-direction checks remove 1–3 edges and 1.5–5% of false pairs, so direction is not the main problem.
- Breaking cycles by vote count removes no false pairs; they rise from 773 to 808.
- Widest-edge rule: these edges are 80% wrong among labeled ones, against 65% across all edges. This row came from a control added after seeing results.
- Limit of repair: removing every labeled-wrong edge (605) still leaves closure precision at .43 with F1 .41, and costs 17.8% of true pairs. Repairs that keep to 50 edges leave direct precision at about .24–.25.

## Limits

- Tune is small and the split is not stratified.
- The gold labels were made independently of the slides, which may understate source inference.
- Only 49% of lectures were downloadable, coverage skews by host, and retrieval was deliberately not tuned. Excerpts include noise such as reference lists and exam pages.
- Slide text is cleared for Muse only. The judge saw names only, and the Muse check is the same model as the proposer.
- The earlier all-dev diagnosis in `scripts/diagnostics/` touched holdout concepts, so the holdout is not fully unseen. That diagnosis also ran before the experiments, and its outputs are in `runs/diagnostics/`.
- Logged `code` hashes come from runs made while `prereq/experiment.py` was still gaining analysis commands. Later edits only added commands and reporting.
- Arm runs totaled about 16.9M Muse tokens against an estimate of about 10M, including one capped run served from cache.
- Two erroneous rows were removed from the log: an empty `votes uncapped` row from a failed shell loop, and a first `repair_controls` row superseded by a later one, whose `new_calls` field wrongly counted cached answers.

## Label agreement and human review

LB1 and LB2 both label 1,595 pairs, about 2% of the labeled pairs. LB1 marks 27 positive, LB2 marks 4, and 3 are shared. Raw agreement is .984 and κ is .19, which is close to positive agreement (.194). The figure rests on 25 disputed pairs, 24 of them LB1 yes and LB2 no, so it says little about label quality overall.

Merged gold counts a pair as positive if either dataset says so, so this asymmetry affects scored precision.

Human review is pending. No result above has been human-checked.

- `runs/kappa_review.csv`: 50 blind pairs, made of all 25 disputes, all 3 joint positives and 22 sampled joint negatives. It answers which dataset is right on disputed pairs. It is a selected set, not a benchmark.
- `runs/review_packet.csv`: 119 blind edges sampled uniformly from what capped, uncapped and source emitted. This one estimates precision, and it has columns for two raters.

Answer keys are local in `runs/private/`.

### Written reviews of both sheets

Two written reviews were scored on 2026-10-31 (`runs/review_ratings.json`, `score_reviews` in `prereq/experiment.py`). The rater was Claude Opus 5.5, as reported by the user, so this is AI review and not the independent human review the project asked for. It is also the same model that designed these experiments and sheets, though not the proposer (Muse) or the frozen judge. An AI rater may share the generator's beliefs, which would inflate agreement with its picks. The sheets did not define "needed" as direct or transitive, and this review rated only 80 of the 119 packet edges.

Agreement sheet, reviewer verdict by label outcome:

| LB1 / LB2 | Pairs | Yes | Defensible | No |
|---|---|---|---|---|
| LB1 yes, LB2 no | 24 | 6 | 4 | 14 |
| LB1 no, LB2 yes | 1 | 0 | 1 | 0 |
| Both yes | 3 | 2 | 1 | 0 |
| Both no | 22 | 0 | 0 | 22 |

Eight of the 14 rejected LB1 positives have Bayes theorem, Conditional probability or Matrix multiplication as the parent, which fits the earlier suspicion that a few LB1 hubs are over-labeled. The disputes are a selected set, so this does not give a label error rate.

Review packet, precision of sampled edges (each arm 40 edges; strict counts only clean yes, lenient adds guideline-dependent items, the floor counts unrated edges as no):

| Arm | Yes / unclear / no / unrated | Strict (rated edges) | Lenient | Floor | Gold, same sample |
|---|---|---|---|---|---|
| Capped | 14 / 9 / 3 / 14 | .54 [.35, .71] | .88 | .35 | 11 of 27 (.41) |
| Uncapped | 10 / 6 / 9 / 15 | .40 [.23, .59] | .64 | .25 | 9 of 28 (.32) |
| Source | 14 / 7 / 9 / 10 | .47 [.30, .64] | .70 | .35 | 6 of 29 (.21) |

The arm intervals overlap, so this does not rank the arms. The packet has no placebo arm, so it cannot revisit the source decision, which also rested on lost recall. Gold and reviewer differ on model-emitted edges: the reviewer calls 17 of 38 rated gold negatives clean prerequisites, against 8 of 14 rated gold positives. Either gold negatives are too strict, or the reviewer is too lenient or counts transitive prerequisites.

No decision above changes on this evidence.

Rating rule for human raters: "Would a course on B have to teach or assume A?" Transitive prerequisites count. A broader topic counts only if a learner must understand it first.

## Next step, not started

Re-rank uncapped candidates using passages that mention both concepts plus which concept comes first in a course, tested against a placebo. Then freeze that choice with the widest-edge rule and score the holdout once.

Update: the pair-level design failed its offline gate, so it has not been run (`docs/offline_checks.md`). Course order was not tested. The holdout was scored without it, and the widest-edge rule failed there (`docs/holdout.md`).
