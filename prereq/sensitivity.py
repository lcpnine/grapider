"""Exploratory offline checks on the tune half. No model calls, and gold and the split do not change.

Usage: python -m prereq.sensitivity comention | cost | duplicates | hubs

Co-mention gate for the pair-level evidence arm, fixed before these checks first ran. All must hold on tune:
- at least 30% of gold-positive pairs and 30% of uncapped candidate edges have a page mentioning both concepts;
- gold-positive pairs co-occur at least 1.5x as often as labeled non-parents, with the interval above 1;
- removing every labeled-wrong edge that has such a page lifts direct F1 by at least 20%, the arm's keep rule.
"""
import itertools
import json
import random
import re
import sys
from collections import defaultdict

import numpy as np

from . import experiment as ex
from . import judge as judge_module
from . import llm
from .llm import Ledger, log_run, pmap
from .materials import Library, name_pattern, tokens
from .pipeline import gold_reachability
from .splits import SPLIT_DIR
from .test_run import git_state

HUBS = ("Bayes theorem", "relational databases", "linear algebra", "natural language processing intro",
        "linguistics basics", "Conditional probability", "Matrix multiplication")
GATE = dict(min_coverage=0.30, min_lift=1.5, min_ceiling_gain=0.20)
THRESHOLDS = dict(false_removed=0.30, true_lost=0.10, recall_lost=0.05)
GENERIC = {"artificial", "tradeoff", "algorithm", "method", "model", "introduction", "basic"}
STOP = {"of", "the", "for", "in"}
RANDOM_DRAWS = 500


def forbid_model_calls():
    """A cache miss must fail here rather than spend tokens."""

    def refuse(*args, **kwargs):
        raise RuntimeError("cache miss: this module makes no model calls")

    llm.OpenAI = refuse
    judge_module.OpenAI = refuse
    judge_module.anthropic.Anthropic = refuse


def record(stage, result):
    log_run("experiments", dict(stage=stage, split="tune", exploratory=True, prompt_version=None, git=git_state(),
                                code=ex.code_hash()), ex.clean(result))
    ex.show(stage, result)


class View:
    """Dev with different gold edges or merged concepts, usable wherever experiment.py expects Dev."""

    def __init__(self, names, edges):
        self.names = names
        self.ids = {n: i for i, n in names.items()}
        labels = defaultdict(list)
        for e in edges:
            if e["prereq_id"] != e["dependent_id"]:
                labels[e["prereq_id"], e["dependent_id"]].append(e["label"])
        self.positive = {p for p, v in labels.items() if 1 in v}
        self.negative = {p for p, v in labels.items() if 1 not in v}
        self.reach, universe = gold_reachability(edges)
        self.universe = {(a, b) for a, b in universe if a != b}


def dev_edges():
    return json.loads((SPLIT_DIR / "system_dev.json").read_text())["edges"]


def pooled_lift(pos_hits, pos_total, neg_hits, neg_total):
    return (pos_hits.sum() / max(pos_total.sum(), 1)) / max(neg_hits.sum() / max(neg_total.sum(), 1), 1e-9)


def lift_interval(pairs_a, pairs_b, hit, targets):
    """Rate of hit in pairs_a over pairs_b, with a 95% interval from resampling targets."""
    index = {t: i for i, t in enumerate(sorted(targets))}
    counts = np.zeros((len(index), 4))
    for pairs, offset in ((pairs_a, 0), (pairs_b, 2)):
        for pair in pairs:
            counts[index[pair[1]], offset] += hit(pair)
            counts[index[pair[1]], offset + 1] += 1
    rng = np.random.default_rng(ex.SEED)
    boots = []
    for _ in range(ex.BOOTSTRAPS):
        s = counts[rng.integers(0, len(counts), len(counts))]
        boots.append(pooled_lift(s[:, 0], s[:, 1], s[:, 2], s[:, 3]))
    low, high = np.percentile(boots, [2.5, 97.5])
    total = counts.sum(axis=0)
    return dict(rate_a=total[0] / max(total[1], 1), rate_b=total[2] / max(total[3], 1),
                lift=pooled_lift(counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3]), low=low, high=high)


def is_nested(a, b):
    """True if one name sits inside the other, so a page naming the longer one names both trivially."""
    x, y = " ".join(tokens(a)), " ".join(tokens(b))
    return f" {x} " in f" {y} " or f" {y} " in f" {x} "


class CoMention:
    """Pages in the whole library naming both concepts. A nested name must also appear outside the longer one."""

    def __init__(self, library, names):
        self.library = library
        self.names = names
        self.patterns = {i: name_pattern(n) for i, n in names.items()}
        self.pages = {i: set(library.matches(n)) for i, n in names.items()}
        self.memo = {}

    def shared(self, a, b):
        pages = self.pages[a] & self.pages[b]
        if not is_nested(self.names[a], self.names[b]):
            return pages
        short, long = (a, b) if len(self.names[a]) <= len(self.names[b]) else (b, a)
        return {p for p in pages if self.patterns[short].search(self.patterns[long].sub(" ", self.library.norm[p]))}

    def count(self, pair):
        key = frozenset(pair)
        if key not in self.memo:
            self.memo[key] = len(self.shared(*pair))
        return self.memo[key]


def prune_ceiling(picks_list, pos, labeled, has_page):
    """Direct F1 if a perfect checker removed every labeled-wrong edge with a page, leaving the rest unchanged."""
    baseline, ceiling, exposed = [], [], []
    for picks in picks_list:
        scored = picks & labeled
        tp = len(scored & pos)
        wrong = {e for e in scored - pos if has_page(e)}
        baseline.append(ex.prf(np.array([[tp, len(scored), len(pos)]], dtype=float))["f1"])
        ceiling.append(ex.prf(np.array([[tp, len(scored) - len(wrong), len(pos)]], dtype=float))["f1"])
        exposed.append(len(wrong) / max(len(scored - pos), 1))
    base, top = float(np.mean(baseline)), float(np.mean(ceiling))
    return dict(direct_f1=base, ceiling_f1=top, gain=(top - base) / max(base, 1e-9), wrong_edges_with_page=float(np.mean(exposed)))


def comention():
    dev = ex.Dev()
    ids = ex.load_split("tune")
    library = Library.load()
    co = CoMention(library, {i: dev.names[i] for i in ids})
    has_page = lambda pair: co.count(pair) > 0
    many = lambda pair: co.count(pair) >= 3
    pos = ex.scope_pairs(dev.positive, ids)
    neg = ex.scope_pairs(dev.negative, ids)
    rate = lambda pairs, hit: sum(map(hit, pairs)) / max(len(pairs), 1)
    pick_sets = [ex.scope_pairs(set(ex.load_picks("uncapped", r)), ids) for r in ex.available("uncapped")]
    candidates = set().union(*pick_sets)
    labeled_candidates = candidates & (pos | neg)
    every_pair = {(a, b) for a, b in itertools.permutations(sorted(ids), 2)} - pos
    rng = random.Random(ex.SEED)
    sample = set(rng.sample(sorted(every_pair), 3000))
    each_run = [dict(edges=len(p), with_page=rate(p, has_page)) for p in pick_sets]
    ceiling = prune_ceiling(pick_sets, pos, pos | neg, has_page)
    vs_labeled = lift_interval(pos, neg, has_page, ids)
    result = dict(
        pages=len(library.pages), concepts=len(ids), concepts_with_a_page=sum(bool(p) for p in co.pages.values()),
        coverage=dict(gold_positive=rate(pos, has_page), gold_positive_3_pages=rate(pos, many),
                      labeled_negative=rate(neg, has_page), labeled_negative_3_pages=rate(neg, many),
                      sampled_all_non_parents=rate(sample, has_page), uncapped_candidates_each_run=each_run,
                      uncapped_candidates_mean=float(np.mean([r["with_page"] for r in each_run])),
                      uncapped_candidates_3_pages=rate(candidates, many)),
        pairs=dict(gold_positive=len(pos), labeled_negative=len(neg), candidates_union=len(candidates), labeled_candidates=len(labeled_candidates)),
        lift_positive_vs_labeled_negative=vs_labeled,
        lift_positive_vs_sampled_non_parents=lift_interval(pos, sample, has_page, ids),
        lift_within_labeled_candidates=lift_interval(labeled_candidates & pos, labeled_candidates - pos, has_page, ids),
        direct_precision_of_candidates=dict(with_page=ex.precision_of({e for e in candidates if has_page(e)}, pos, pos | neg),
                                            without_page=ex.precision_of({e for e in candidates if not has_page(e)}, pos, pos | neg)),
        nested_candidates=sum(is_nested(dev.names[a], dev.names[b]) for a, b in candidates),
        broadest_names={dev.names[i]: len(co.pages[i]) for i in sorted(ids, key=lambda i: -len(co.pages[i]))[:12]},
        prune_ceiling=ceiling,
        gate=GATE,
    )
    coverage = result["coverage"]
    checks = dict(gold_coverage=coverage["gold_positive"] >= GATE["min_coverage"],
                  candidate_coverage=coverage["uncapped_candidates_mean"] >= GATE["min_coverage"],
                  lift=bool(vs_labeled["lift"] >= GATE["min_lift"] and vs_labeled["low"] > 1),
                  ceiling=ceiling["gain"] >= GATE["min_ceiling_gain"])
    result["gate_checks"] = checks
    result["gate_passed"] = all(checks.values())
    record("exploratory_comention", result)


def cost():
    """Calls the pair-level arm would make over all dev. Uses names, picks and cached check sizes, and no labels."""
    forbid_model_calls()
    dev = ex.Dev()
    library = Library.load()
    co = CoMention(library, dev.names)
    runs = [set(ex.load_picks("uncapped", r)) for r in ex.available("uncapped")]
    calls_per_arm = sum(co.count(e) > 0 for run in runs for e in run)
    passages = ex.real_passages(dev, library)
    ledger = Ledger()
    for r in ex.available("uncapped"):
        graph, _ = ex.assemble(dev, ex.load_picks("uncapped", r))
        bits = ex.Bits(graph, dev, ex.load_split("tune"))
        for _, (a, b) in bits.impact()[:ex.CHECK_BUDGET]:
            ex.ask_check(dev.names[a], dev.names[b], passages, ledger)
    per_call = ledger.tokens / ledger.calls
    tokens_two_arms = 2 * calls_per_arm * per_call
    record("exploratory_cost", dict(edges_all_runs=sum(map(len, runs)), calls_per_arm=calls_per_arm, tokens_per_cached_check=per_call,
                                    tokens_real_plus_placebo=tokens_two_arms, with_past_overrun_1_7x=1.7 * tokens_two_arms))


def compact_forms(name):
    """Words of a name without stop words or plurals."""
    words = []
    for word in tokens(name):
        if word == "s" or word in STOP:
            continue
        words.append(word[:-1] if word.endswith("s") and not word.endswith(("ss", "is", "us")) and len(word) > 3 else word)
    return words


def is_abbreviation(short, long):
    """Seq2seq for sequence to sequence: each piece begins the matching word, and 2 stands for to."""
    pieces = re.findall(r"[a-z]+|\d+", short)
    words = ["2" if w == "to" else w for w in long]
    return len(pieces) == len(words) and len(pieces) > 1 and all(w.startswith(p) for p, w in zip(pieces, words))


def same_concept(a, b):
    """Rule for duplicates: same words apart from generic ones, the same letters, or an abbreviation or initials."""
    raw_a, raw_b = [w for w in tokens(a) if w != "s"], [w for w in tokens(b) if w != "s"]
    x, y = compact_forms(a), compact_forms(b)
    two_topics = "and" in raw_a or "and" in raw_b
    if not two_topics and {w for w in x if w not in GENERIC} == {w for w in y if w not in GENERIC} and any(w not in GENERIC for w in x):
        return "same words"
    if "".join(x) == "".join(y):
        return "same letters"
    for short, long, raw_long in ((x, y, raw_b), (y, x, raw_a)):
        if len(short) == 1 and len(raw_long) > 1:
            initials = "".join(w[0] for w in raw_long)
            if short[0] == initials and len(initials) > 1:
                return "initials"
            if is_abbreviation(short[0], raw_long):
                return "abbreviation"
    return None


def duplicate_pairs(names):
    found = []
    for a, b in itertools.combinations(sorted(names), 2):
        rule = same_concept(names[a], names[b])
        if rule:
            found.append((a, b, rule))
    return found


def merge_maps(pairs, tune):
    """Predictions merge on any pair touching tune. Gold merges only when every member is in tune, so no holdout label enters."""
    groups = {}
    for a, b, _ in pairs:
        joined = groups.get(a, {a}) | groups.get(b, {b})
        for member in joined:
            groups[member] = joined
    pick_map, gold_map = {}, {}
    for members in {frozenset(g) for g in groups.values()}:
        inside = sorted(m for m in members if m in tune)
        if not inside:
            continue
        for member in members:
            pick_map[member] = inside[0]
        for member in inside:
            gold_map[member] = inside[0]
    return pick_map, gold_map


def map_picks(picks, mapping):
    out = {}
    for (a, b), why in picks.items():
        pair = mapping.get(a, a), mapping.get(b, b)
        if pair[0] != pair[1]:
            out.setdefault(pair, why)
    return out


def merged_view(dev, edges, gold_map, pick_map):
    gone = {i for i, rep in pick_map.items() if i != rep}
    names = {i: n for i, n in dev.names.items() if i not in gone}
    moved = [dict(e, prereq_id=gold_map.get(e["prereq_id"], e["prereq_id"]), dependent_id=gold_map.get(e["dependent_id"], e["dependent_id"])) for e in edges]
    return View(names, moved)


def arm_counts(dev, arm, ids, targets=None, mapping=None):
    runs = [ex.measure(dev, map_picks(ex.load_picks(arm, r), mapping or {}), ids, targets) for r in ex.available(arm)]
    return {kind: np.mean([run[kind] for run in runs], axis=0) for kind in ("direct", "reach")}


def compare(view, ids, targets, arms, pairs, mapping=None):
    """Direct and closure scores per arm and paired gains."""
    counts = {arm: arm_counts(view, arm, ids, targets, mapping) for arm in arms}
    scores = {arm: {kind: ex.prf(c[kind]) for kind in c} for arm, c in counts.items()}
    gains = {f"{a} vs {b}": {kind: ex.paired_gain(counts[a][kind], counts[b][kind])["f1"] for kind in ("direct", "reach")} for a, b in pairs}
    return dict(scores=scores, gains=gains)


CAP = (("capped", "uncapped"), None, [("uncapped", "capped")])
SOURCE_PAIRS = [("source", "uncapped"), ("source", "placebo")]
SOURCE_ARMS = ("uncapped", "source", "placebo")


def duplicates():
    dev = ex.Dev()
    edges = dev_edges()
    tune = ex.load_split("tune")
    pairs = duplicate_pairs(dev.names)
    pick_map, gold_map = merge_maps(pairs, tune)
    view = merged_view(dev, edges, gold_map, pick_map)
    ids = {gold_map.get(i, i) for i in tune}
    where = lambda a, b: "both tune" if a in tune and b in tune else "neither in tune" if a not in tune and b not in tune else "straddles"
    listed = [dict(a=dev.names[a], b=dev.names[b], rule=rule, where=where(a, b)) for a, b, rule in pairs]
    covered = ex.covered_targets(dev, tune)
    baseline = dict(cap=compare(dev, tune, None, CAP[0], CAP[2]), source=compare(dev, tune, covered, SOURCE_ARMS, SOURCE_PAIRS))
    merged_covered = {gold_map.get(t, t) for t in covered}
    merged = dict(cap=compare(view, ids, None, CAP[0], CAP[2], pick_map), source=compare(view, ids, merged_covered, SOURCE_ARMS, SOURCE_PAIRS, pick_map))
    pos_before, pos_after = ex.scope_pairs(dev.positive, tune), ex.scope_pairs(view.positive, ids)
    record("exploratory_duplicates", dict(pairs=listed, gold_positive_before=len(pos_before), gold_positive_after=len(pos_after),
                                          baseline=baseline, merged=merged, not_rescored="repair rows: merging changes edges the checkers never saw"))


def without_hubs(dev, edges, positives_only=True):
    hubs = {dev.ids[n] for n in HUBS}
    kept = [e for e in edges if not (e["prereq_id"] in hubs and (e["label"] == 1 or not positives_only))]
    return View(dev.names, kept)


def checker_answers(dev, top, passages):
    ledger = Ledger()
    name = dev.names
    judge, _ = ex.judge_checks(dev, top)
    muse = pmap(lambda e: ex.ask_check(name[e[0]], name[e[1]], passages, ledger), top, workers=12)
    return judge, muse


def repair_rows(view, ids, passages, dev):
    """The experiment 3 rows under view's gold. Edges and checker answers are unchanged, because no labels pick them."""
    rows = defaultdict(list)
    reach_size = len(ex.scope_pairs(view.reach, ids))
    pos = ex.scope_pairs(view.positive, ids)
    for r in ex.available("uncapped"):
        graph, _ = ex.assemble(dev, ex.load_picks("uncapped", r))
        bits = ex.Bits(graph, view, ids)
        base_tp, base_fp = bits.score()
        direct = bits.edges & pos
        top = [e for _, e in bits.impact()[:ex.CHECK_BUDGET]]
        judge, muse = checker_answers(dev, top, passages)

        def effect(removed):
            tp, fp = bits.score(removed)
            return dict(false_removed=(base_fp - fp) / max(base_fp, 1), true_lost=(base_tp - tp) / max(base_tp, 1),
                        recall_lost=len(direct & removed) / max(len(direct), 1), closure_f1=ex.closure_f1(tp, fp, reach_size))

        rows["none"].append(effect(set()))
        rows["judge no"].append(effect({e for e, yes in zip(top, judge) if not yes}))
        rows["muse+excerpts no"].append(effect({e for e, yes in zip(top, muse) if not yes}))
        rows["50 widest"].append(effect(set(top)))
        rng = random.Random(ex.SEED + r)
        every = sorted(bits.edges)
        draws = [effect(set(rng.sample(every, ex.CHECK_BUDGET))) for _ in range(RANDOM_DRAWS)]
        rows["50 random"].append({k: float(np.mean([d[k] for d in draws])) for k in draws[0]})
        curve = ex.oracle_curve(bits, view)
        rows["gold limit 50"].append(effect(curve.get(50, curve["all wrong"])))
    summary = {label: {k: float(np.mean([g[k] for g in group])) for k in group[0]} for label, group in rows.items()}
    for row in summary.values():
        row["passes"] = bool(row["false_removed"] >= THRESHOLDS["false_removed"] and row["true_lost"] < THRESHOLDS["true_lost"]
                             and row["recall_lost"] < THRESHOLDS["recall_lost"])
    return summary


def hubs():
    forbid_model_calls()
    dev = ex.Dev()
    edges = dev_edges()
    ids = ex.load_split("tune")
    covered = ex.covered_targets(dev, ids)
    passages = ex.real_passages(dev, Library.load())
    views = dict(as_labeled=dev, hub_positives_removed=without_hubs(dev, edges), all_hub_parent_pairs_removed=without_hubs(dev, edges, positives_only=False))
    result = dict(hubs=HUBS, thresholds=THRESHOLDS)
    for label, view in views.items():
        result[label] = dict(tune_positives=len(ex.scope_pairs(view.positive, ids)),
                             cap=compare(view, ids, None, CAP[0], CAP[2]),
                             source=compare(view, ids, covered, SOURCE_ARMS, SOURCE_PAIRS),
                             repair=repair_rows(view, ids, passages, dev))
    record("exploratory_hubs", result)


def main():
    dict(comention=comention, cost=cost, duplicates=duplicates, hubs=hubs)[sys.argv[1]]()


if __name__ == "__main__":
    main()
