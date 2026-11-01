# Cap removal in the pipeline

Date: 2026-10-31. Generation no longer limits how many parents a concept can have. This was a design decision, not a score gain: on the tune half, experiment 1 showed recall up, precision down and F1 within noise (`docs/experiments_1_3.md`).

## What changed

- `DEFAULT_EDGE_PROMPT` is now `e4_uncapped`, with the same text as the uncapped arm.
- `MAX_PARENTS`, the `too_many_parents` validator and `enforce_parent_limit` are gone.
- The old prompts `e1`–`e3` keep their exact text and a 3-parent cap (`LEGACY_CAP`), so earlier results can still be reproduced.
- The experiment's `capped` arm is pinned to `e2_strict`, and a test checks the prompt hashes against the logged ones.
- The earlier one-shot test results used the capped pipeline. Any new test data would use the uncapped one.

Replay check, with model calls forbidden: all 6 capped and uncapped runs match their saved picks. The pipeline reproduces replicate 0 of both arms.

## Civics re-check (dev)

Each method ran once on the 2008 to 2025 revision, so there are no intervals. HAM and the test data were not touched. The capped column is the last logged capped run. New spend was 3.14M Muse tokens (estimate: about 3M).

| Revision agent | Capped | Uncapped |
|---|---|---|
| Edges | 202 | 258 |
| Most parents on one concept | 3 | 9 |
| Bridged edges (not proposed by the model) | 14 | 30 |
| Affected-node F1 | .984 | .984 |
| Unaffected edges and orderings kept | 100% | 100% |
| Stale nodes | 0 | 0 |
| Cycles left | 0 | 0 |
| Orphans | 22 | 12 |
| Frozen judge says yes, 150 sampled edges | .627 [.547, .700] | .547 [.467, .624] |

Plan violations against that run's own full rebuild, for a learner 40% through the old plan. Each column uses its own rebuild, so the reference differs between them.

| Share of reference orderings violated | Capped | Uncapped |
|---|---|---|
| Revision agent | 23.5% (121 of 514) | 15.2% (132 of 869) |
| Append-only | 27.4% (141 of 514) | 15.9% (138 of 869) |
| Rebuild with candidate list reversed (noise floor) | 7.5% (42 of 561) | 18.4% (180 of 979) |

What this shows:

- **The revision agent's guarantees held.** Change detection, stability and validity are unchanged.
- **Judge agreement fell in all four methods**, by .05 to .11. For the agent the intervals overlap.
- **Rebuilds became less stable.** Reversing the candidate list now changes orderings as much as a revision does. The agent's lower violation rate therefore does not show it improved, because its reference is noisier and larger.
- **The agent's advantage over append-only on plan violations is gone** (15.2% vs 15.9%). It still keeps no stale nodes, against 51 for append-only, and finds more of the affected nodes (.984 vs .834).
- **Bridged edges doubled**, because nothing trims them any more. They are now 12% of the revised graph. Their quality was not measured.

Not verified: `scripts/diagnostics/` was updated for the new imports and compiles, but it was not rerun, because it reads holdout labels.
