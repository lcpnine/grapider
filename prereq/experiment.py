"""Controlled dev experiments: cap removal, source-grounded inference, graph repair. Reads no test data.

Every arm builds a graph over all dev concepts. Choices use pairs inside the tune half, and the holdout half
is scored once, after those choices are written to exp_freeze.json.

Usage: python -m prereq.experiment split | arm <name> <replicate> | report cap|source | signal | repair <arm>
       | controls <arm> | uptake | votes <arm> | holdout | packet | disputes | score_reviews
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from .dag import MAX_PARENTS, PrereqDAG
from .judge import PROMPTS as JUDGE_PROMPTS
from .judge import predict
from .llm import MUSE_MODEL, RUNS, Ledger, ask_json, log_run, pmap
from .materials import Library, name_pattern, normal
from .pipeline import DEFAULT_EDGE_PROMPT, EDGE_OUTPUT, EDGE_PROMPTS, break_cycles, gold_reachability
from .revision import FROZEN
from .schema import canon
from .splits import SPLIT_DIR
from .test_run import git_state

SPLIT = SPLIT_DIR / "exp_split.json"
FREEZE = SPLIT_DIR / "exp_freeze.json"
PRIVATE = RUNS / "private"
PICKS = PRIVATE / "exp_picks"
LOG = RUNS / "experiments.jsonl"
SEED = 20261031
REPLICATES = 3
CHECK_BUDGET = 50
ORACLE_BUDGETS = (10, 25, 50, 100)
BOOTSTRAPS = 2000

UNCAPPED = ("You build a prerequisite graph. Given a target concept and a list of concepts, pick every concept "
            "from the list that a learner must understand before the target. Being related is not enough, and "
            "do not pick concepts that build on the target. Many foundational concepts have no prerequisites "
            "in the list. " + EDGE_OUTPUT)
SOURCE = UNCAPPED + (" Excerpts from course materials retrieved for the target are included. "
                     "Use them as evidence of what the target builds on.")
CHECK = JUDGE_PROMPTS["v1_direct"] + " Course excerpts are included. Use them as evidence."
ARMS = {
    "capped": dict(system=EDGE_PROMPTS[DEFAULT_EDGE_PROMPT], cap=MAX_PARENTS, passages=None),
    "uncapped": dict(system=UNCAPPED, cap=None, passages=None),
    "source": dict(system=SOURCE, cap=None, passages="real"),
    "placebo": dict(system=SOURCE, cap=None, passages="placebo"),
}


class Dev:
    """Dev concepts and labels. A pair labeled 1 by either dataset counts as positive."""

    def __init__(self):
        split = json.loads((SPLIT_DIR / "system_dev.json").read_text())
        self.names = {c["id"]: c["name"] for c in split["concepts"]}
        self.ids = {n: i for i, n in self.names.items()}
        labels = defaultdict(list)
        for e in split["edges"]:
            labels[e["prereq_id"], e["dependent_id"]].append(e["label"])
        self.positive = {p for p, v in labels.items() if 1 in v}
        self.negative = {p for p, v in labels.items() if 1 not in v}
        self.reach, self.universe = gold_reachability(split["edges"])


def make_split():
    if SPLIT.exists():
        raise FileExistsError(f"{SPLIT} exists; the split is frozen")
    ids = sorted(Dev().names)
    random.Random(SEED).shuffle(ids)
    half = len(ids) // 2
    SPLIT.write_text(json.dumps(dict(seed=SEED, tune=sorted(ids[:half]), holdout=sorted(ids[half:])), indent=0))


def load_split(name):
    return set(json.loads(SPLIT.read_text())[name])


def excerpt_block(passages):
    return "\n".join(f'[{i}] {p["title"]}: {p["text"]}' for i, p in enumerate(passages, 1))


def real_passages(dev, library):
    return {n: library.passages(n) for n in dev.names.values()}


def placebo_passages(real, seed=SEED):
    """For each target, equally long excerpts from a concept that shares no mention with it. Uses no labels."""
    rng = random.Random(seed)
    covered = sorted(n for n, p in real.items() if p)
    text = {n: normal(" ".join(p["text"] for p in real[n])) for n in covered}
    patterns = {n: name_pattern(n) for n in covered}
    placebo = {}
    for target in covered:
        length = sum(len(p["text"]) for p in real[target])
        pool = covered[:]
        rng.shuffle(pool)
        fits = [d for d in pool if d != target and not patterns[target].search(text[d]) and not patterns[d].search(text[target])]
        donor = next((d for d in fits if sum(len(p["text"]) for p in real[d]) >= length), max(fits, key=lambda d: len(text[d])))
        trimmed, left = [], length
        for p in real[donor]:
            if left > 0:
                trimmed.append(dict(title=p["title"], text=p["text"][:left]))
                left -= len(trimmed[-1]["text"])
        placebo[target] = trimmed
    return placebo


def arm_passages(arm, dev):
    kind = ARMS[arm]["passages"]
    if kind is None:
        return {}
    real = real_passages(dev, Library.load())
    return real if kind == "real" else placebo_passages(real)


def pick_parents(target, names, config, passages, ledger, replicate):
    """Parents of one target. A target with no excerpts gets the same names-only call as the uncapped arm."""
    others = sorted(n for n in names if n != target)
    listing = "Concepts:\n" + "\n".join(f"- {n}" for n in others)
    excerpts = passages.get(target)
    if excerpts:
        out = ask_json(config["system"], f'Target: "{target}"\nCourse excerpts:\n{excerpt_block(excerpts)}\n{listing}',
                       ledger, replicate=replicate, private=True)
    else:
        system = config["system"] if config["passages"] is None else UNCAPPED
        out = ask_json(system, f'Target: "{target}"\n{listing}', ledger, replicate=replicate)
    known = {canon(n): n for n in others}
    picks, unknown = {}, 0
    for p in out.get("prereqs", []):
        name = known.get(canon(str(p.get("name", ""))))
        if name:
            picks.setdefault(name, " ".join(str(p.get("reason") or "selected by model").split()))
        else:
            unknown += 1
    chosen = list(picks.items())
    return chosen[: config["cap"]] if config["cap"] else chosen, unknown


def picks_path(arm, replicate):
    return PICKS / f"{arm}_{replicate}.json"


def run_arm(arm, replicate):
    dev = Dev()
    config = ARMS[arm]
    passages = arm_passages(arm, dev)
    ledger = Ledger()
    names = sorted(dev.names.values())
    results = pmap(lambda t: pick_parents(t, names, config, passages, ledger, replicate), names, workers=12)
    edges = [[dev.ids[p], dev.ids[t], why] for t, (chosen, _) in zip(names, results) for p, why in chosen]
    PICKS.mkdir(parents=True, exist_ok=True)
    picks_path(arm, replicate).write_text(json.dumps(edges))
    result = dict(edges=len(edges), unknown_names=sum(u for _, u in results), with_excerpts=sum(bool(passages.get(t)) for t in names),
                  **ledger.to_dict())
    log_run("experiments", dict(stage="arm", arm=arm, replicate=replicate, prompt_version=hash_text(config["system"]),
                                passages=config["passages"], model=MUSE_MODEL, git=git_state(), code=code_hash()), result)
    print(arm, replicate, result, flush=True)


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def code_hash():
    return hash_text(Path(__file__).read_text())


def load_picks(arm, replicate):
    return {(a, b): why for a, b, why in json.loads(picks_path(arm, replicate).read_text())}


def assemble(dev, picks, votes=None):
    """Return (acyclic graph before reduction, removed cycle edges). votes makes cycles lose their weakest edge."""
    dag = PrereqDAG()
    for i, name in dev.names.items():
        dag.add_node(i, name, ["dev"])
    for (a, b), why in sorted(picks.items()):
        dag.add_edge(a, b, why or "selected by model")
    judge = None
    if votes is not None:
        judge = lambda cycle: min(cycle, key=lambda e: (votes.get(e, 0), dag.g.out_degree(e[0]), e))
    removed = break_cycles(dag, judge)
    return dag.g, removed


def closure_of(graph):
    return {(u, v) for u, v in nx.transitive_closure(graph, reflexive=False).edges if u != v}


def per_target(pred, positives, scored, targets):
    """Counts per target: true positives, predicted in scope, positives."""
    rows = {t: [0, 0, 0] for t in targets}
    for a, b in pred & scored:
        if b in rows:
            rows[b][1] += 1
            if (a, b) in positives:
                rows[b][0] += 1
    for a, b in positives & scored:
        if b in rows:
            rows[b][2] += 1
    return np.array([rows[t] for t in targets], dtype=float)


def prf(counts):
    tp, predicted, positive = counts.sum(axis=0)
    p = tp / predicted if predicted else 0.0
    r = tp / positive if positive else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if tp else 0.0, scored=int(predicted))


def scope_pairs(pairs, ids):
    return {p for p in pairs if p[0] in ids and p[1] in ids}


def measure(dev, picks, ids, targets=None):
    """Direct and closure counts for one run, all inside one split half."""
    targets = sorted(targets if targets is not None else ids)
    pos = scope_pairs(dev.positive, ids)
    labeled = pos | scope_pairs(dev.negative, ids)
    reach = scope_pairs(dev.reach, ids)
    universe = scope_pairs(dev.universe, ids)
    graph, removed = assemble(dev, picks)
    closure = closure_of(graph)
    return dict(
        direct=per_target(set(picks), pos, labeled, targets),
        reach=per_target(closure, reach, universe, targets),
        raw=per_target(closure, pos, labeled, targets),
        extras=dict(edges=len(picks), cycle_edges_removed=len(removed), closure_scored=len(closure & universe),
                    unlabeled_picks=len(scope_pairs(set(picks), ids) - labeled),
                    per_target=len(picks) / len(dev.names)),
    )


def available(arm):
    return [r for r in range(REPLICATES) if picks_path(arm, r).exists()]


def arm_summary(dev, arm, ids, targets=None):
    runs = [measure(dev, load_picks(arm, r), ids, targets) for r in available(arm)]
    summary = {}
    for kind in ("direct", "reach", "raw"):
        mean = prf(np.mean([run[kind] for run in runs], axis=0))
        each = [prf(run[kind])["f1"] for run in runs]
        summary[kind] = {**mean, "f1_each_run": each}
    summary["extras"] = {k: float(np.mean([run["extras"][k] for run in runs])) for k in runs[0]["extras"]}
    summary["counts"] = {kind: np.mean([run[kind] for run in runs], axis=0) for kind in ("direct", "reach", "raw")}
    return summary


def paired_gain(a, b, seed=SEED):
    """F1 and precision/recall gain of a over b, with 95% intervals from resampling targets."""
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = []
    for _ in range(BOOTSTRAPS):
        s = rng.integers(0, n, n)
        x, y = prf(a[s]), prf(b[s])
        boots.append([x[k] - y[k] for k in ("f1", "precision", "recall")])
    low, high = np.percentile(boots, [2.5, 97.5], axis=0)
    x, y = prf(a), prf(b)
    return {k: dict(gain=x[k] - y[k], low=low[i], high=high[i]) for i, k in enumerate(("f1", "precision", "recall"))}


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != "counts"}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return round(float(value), 4) if isinstance(value, (float, np.floating)) else value


def show(label, value):
    print(label, json.dumps(clean(value)), flush=True)


def covered_targets(dev, ids):
    library = Library.load()
    return {i for i in ids if library.passages(dev.names[i])}


def report(name):
    dev = Dev()
    ids = load_split("tune")
    if name == "cap":
        arms, targets, pairs = ("capped", "uncapped"), None, [("uncapped", "capped")]
    else:
        arms, targets, pairs = ("uncapped", "source", "placebo"), covered_targets(dev, ids), [("source", "uncapped"), ("source", "placebo"), ("placebo", "uncapped")]
    results = {arm: arm_summary(dev, arm, ids, targets) for arm in arms}
    for arm, summary in results.items():
        show(arm, summary)
    gains = {f"{a} vs {b}": {kind: paired_gain(results[a]["counts"][kind], results[b]["counts"][kind]) for kind in ("direct", "reach", "raw")} for a, b in pairs}
    for label, gain in gains.items():
        show(label, gain)
    log_run("experiments", dict(stage=f"report_{name}", split="tune", targets=len(targets) if targets else len(ids),
                                prompt_version="see arms", git=git_state(), code=code_hash()),
            clean(dict(arms=results, gains=gains)))


def signal():
    """Before spending calls: do gold parents appear in a target's passages more than non-parents? Tune pairs only."""
    dev = Dev()
    library = Library.load()
    ids = load_split("tune")
    real = {i: library.passages(dev.names[i]) for i in dev.names}
    every = len(dev.names)
    covered = [i for i in dev.names if real[i]]
    deep = [i for i in dev.names if len(real[i]) >= 3]
    pos = scope_pairs(dev.positive, ids)
    neg = scope_pairs(dev.negative, ids)
    patterns = {i: name_pattern(dev.names[i]) for i in dev.names}
    text = {i: normal(" ".join(p["text"] for p in real[i])) for i in covered}
    mention = lambda pair: pair[1] in text and bool(patterns[pair[0]] and patterns[pair[0]].search(text[pair[1]]))
    usable_pos = [p for p in pos if p[1] in text]
    usable_neg = [p for p in neg if p[1] in text]
    hit_pos = sum(map(mention, usable_pos))
    hit_neg = sum(map(mention, usable_neg))
    p = hit_pos / max(hit_pos + hit_neg, 1)
    r = hit_pos / max(len(usable_pos), 1)
    result = dict(concepts=every, covered=len(covered), coverage=len(covered) / every, with_3_passages=len(deep),
                  tune_positives=len(pos), tune_positives_with_target_passages=len(usable_pos),
                  tune_positive_coverage=len(usable_pos) / max(len(pos), 1),
                  mention_rate_parents=hit_pos / max(len(usable_pos), 1), mention_rate_others=hit_neg / max(len(usable_neg), 1),
                  lift=(hit_pos / max(len(usable_pos), 1)) / max(hit_neg / max(len(usable_neg), 1), 1e-9),
                  mention_rule=dict(precision=p, recall=r, f1=2 * p * r / (p + r) if hit_pos else 0.0))
    log_run("experiments", dict(stage="signal", split="tune", prompt_version=None, git=git_state(), code=code_hash()), clean(result))
    show("signal", result)


class Bits:
    """Fast closure with integer bitsets, for trying many edge removals."""

    def __init__(self, graph, dev, ids):
        self.nodes = list(nx.topological_sort(graph))
        self.index = {n: i for i, n in enumerate(self.nodes)}
        self.edges = {(u, v) for u, v in graph.edges}
        reach, universe = scope_pairs(dev.reach, ids), scope_pairs(dev.universe, ids)
        self.true = defaultdict(int)
        self.false = defaultdict(int)
        for a, b in universe:
            if a in self.index and b in self.index:
                target = self.true if (a, b) in reach else self.false
                target[a] |= 1 << self.index[b]

    def closure(self, skip=frozenset()):
        children = defaultdict(list)
        for u, v in self.edges:
            if (u, v) not in skip:
                children[u].append(v)
        desc = {}
        for n in reversed(self.nodes):
            bits = 0
            for child in children[n]:
                bits |= desc[child] | (1 << self.index[child])
            desc[n] = bits
        return desc

    def pairs(self, skip=frozenset()):
        return sum(b.bit_count() for b in self.closure(skip).values())

    def score(self, skip=frozenset()):
        desc = self.closure(skip)
        return (sum((desc[n] & self.true[n]).bit_count() for n in self.nodes),
                sum((desc[n] & self.false[n]).bit_count() for n in self.nodes))

    def impact(self, skip=frozenset(), only=None):
        """Closure pairs each edge alone supports, largest first."""
        base = self.pairs(skip)
        pool = (only if only is not None else self.edges) - skip
        return sorted(((base - self.pairs(skip | {e}), e) for e in pool), key=lambda x: (-x[0], x[1]))


def repair_row(bits, dev, ids, direct_base, removed):
    tp, fp = bits.score(removed)
    base_tp, base_fp = bits.score()
    pos = scope_pairs(dev.positive, ids)
    labeled = pos | scope_pairs(dev.negative, ids)
    kept = (bits.edges - removed) & labeled
    hits = len(kept & pos)
    return dict(removed=len(removed), true_pairs=tp, false_pairs=fp,
                false_removed_share=(base_fp - fp) / max(base_fp, 1), true_lost_share=(base_tp - tp) / max(base_tp, 1),
                direct_hits=hits, direct_recall_lost=(direct_base - hits) / max(direct_base, 1),
                closure=dict(precision=tp / max(tp + fp, 1), recall=tp / max(len(scope_pairs(dev.reach, ids)), 1)),
                direct_precision=hits / max(len(kept), 1))


def oracle_curve(bits, dev):
    """Upper limit: remove edges gold calls wrong, widest first. Labels from all of dev pick edges; only tune is scored."""
    wrong = {e for e in bits.edges if e not in dev.reach and e in dev.universe}
    removed, curve = set(), {}
    for step in range(1, max(ORACLE_BUDGETS) + 1):
        ranked = bits.impact(removed, wrong)
        if not ranked or ranked[0][0] == 0:
            break
        removed.add(ranked[0][1])
        if step in ORACLE_BUDGETS:
            curve[step] = set(removed)
    curve["all wrong"] = wrong
    return curve


def ask_check(a, b, passages, ledger):
    """Muse with excerpts on both concepts. Not independent of the proposer, but it may use slide text."""
    user = (f'Concept A: "{a}"\nConcept B: "{b}"\nCourse excerpts on A:\n{excerpt_block(passages.get(a, [])[:3])}\n'
            f'Course excerpts on B:\n{excerpt_block(passages.get(b, [])[:3])}\nIs A a prerequisite of B?')
    return bool(ask_json(CHECK, user, ledger, private=True).get("requires"))


def judge_checks(dev, edges):
    """Frozen judge, names only: slide text is cleared for Muse and not for this provider."""
    frozen = json.loads(FROZEN.read_text())
    named = [(dev.names[a], dev.names[b]) for a, b in edges]
    forward = predict(frozen["model"], frozen["prompt_version"], named)
    backward = predict(frozen["model"], frozen["prompt_version"], [(b, a) for a, b in named])
    return forward, backward


def repair(arm):
    dev = Dev()
    ids = load_split("tune")
    passages = real_passages(dev, Library.load())
    reps = available(arm)
    all_votes = Counter(e for r in reps for e in load_picks(arm, r))
    rows = defaultdict(list)
    for r in reps:
        picks = load_picks(arm, r)
        graph, _ = assemble(dev, picks)
        bits = Bits(graph, dev, ids)
        pos = scope_pairs(dev.positive, ids)
        direct_base = len(bits.edges & pos)
        add = lambda label, removed: rows[label].append(repair_row(bits, dev, ids, direct_base, removed))
        add("baseline", set())
        for budget, removed in oracle_curve(bits, dev).items():
            add(f"oracle {budget}", removed)
        voted, _ = assemble(dev, picks, all_votes)
        voted_bits = Bits(voted, dev, ids)
        rows["cycles by votes"].append(repair_row(voted_bits, dev, ids, direct_base, set()))
        top = [e for gain, e in bits.impact()[:CHECK_BUDGET]]
        forward, backward = judge_checks(dev, top)
        add("judge no", {e for e, f in zip(top, forward) if not f})
        add("judge no and reverse yes", {e for e, f, g in zip(top, forward, backward) if not f and g})
        ledger = Ledger()
        name = dev.names
        forward = pmap(lambda e: ask_check(name[e[0]], name[e[1]], passages, ledger), top, workers=12)
        backward = pmap(lambda e: ask_check(name[e[1]], name[e[0]], passages, ledger), top, workers=12)
        add("muse+source no", {e for e, f in zip(top, forward) if not f})
        add("muse+source no and reverse yes", {e for e, f, g in zip(top, forward, backward) if not f and g})
        top_true = sum((e in dev.reach) for e in top)
        rows["checked edges"].append(dict(checked=len(top), in_gold_closure=top_true, in_universe=sum(e in dev.universe for e in top)))
    summary = {}
    for label, group in rows.items():
        keys = [k for k, v in group[0].items() if isinstance(v, (int, float))]
        summary[label] = {k: float(np.mean([g[k] for g in group])) for k in keys}
    log_run("experiments", dict(stage="repair", arm=arm, split="tune", prompt_version=None, git=git_state(), code=code_hash()), clean(summary))
    for label, row in summary.items():
        show(label, row)


def votes(arm):
    """Exploratory: precision by how many replicates chose an edge, and the majority graph."""
    dev = Dev()
    ids = load_split("tune")
    reps = available(arm)
    counted = Counter(e for r in reps for e in load_picks(arm, r))
    pos = scope_pairs(dev.positive, ids)
    labeled = pos | scope_pairs(dev.negative, ids)
    bins = {}
    for k in range(1, len(reps) + 1):
        edges = {e for e, n in counted.items() if n == k}
        scored = edges & labeled
        bins[k] = dict(edges=len(edges), scored=len(scored), direct_precision=len(scored & pos) / max(len(scored), 1), hits=len(scored & pos))
    runs = [set(load_picks(arm, r)) for r in reps]
    overlap = [len(a & b) / max(len(a | b), 1) for i, a in enumerate(runs) for b in runs[i + 1:]]
    majority = {e: "vote" for e, n in counted.items() if n >= 2}
    run = measure(dev, majority, ids)
    result = dict(by_votes=bins, run_overlap=overlap, majority=dict(direct=prf(run["direct"]), reach=prf(run["reach"]), raw=prf(run["raw"]), extras=run["extras"]))
    log_run("experiments", dict(stage="votes", arm=arm, split="tune", prompt_version=None, git=git_state(), code=code_hash()), clean(result))
    show(arm, result)


def closure_f1(tp, fp, positives):
    precision, recall = tp / max(tp + fp, 1), tp / max(positives, 1)
    return 2 * precision * recall / (precision + recall) if tp else 0.0


def controls(arm, draws=500):
    """Are the checkers better than removing the same number of top edges blindly? Uses cached check answers."""
    dev = Dev()
    ids = load_split("tune")
    passages = real_passages(dev, Library.load())
    name = dev.names
    reach_size = len(scope_pairs(dev.reach, ids))
    ledger = Ledger()
    rows = defaultdict(list)
    for r in available(arm):
        graph, _ = assemble(dev, load_picks(arm, r))
        bits = Bits(graph, dev, ids)
        base_tp, base_fp = bits.score()
        top = [e for _, e in bits.impact()[:CHECK_BUDGET]]
        judge, _ = judge_checks(dev, top)
        muse = pmap(lambda e: ask_check(name[e[0]], name[e[1]], passages, ledger), top, workers=12)
        rng = random.Random(SEED + r)

        def effect(removed):
            tp, fp = bits.score(removed)
            return dict(removed=len(removed), false_removed_share=(base_fp - fp) / max(base_fp, 1),
                        true_lost_share=(base_tp - tp) / max(base_tp, 1), closure_f1=closure_f1(tp, fp, reach_size))

        pos = scope_pairs(dev.positive, ids)
        labeled = pos | scope_pairs(dev.negative, ids)
        direct_base = bits.edges & pos

        def widest(removed):
            kept = (bits.edges - removed) & labeled
            tp, fp = bits.score(removed)
            return dict(**effect(removed), closure_precision=tp / max(tp + fp, 1), closure_recall=tp / max(reach_size, 1),
                        direct_recall_lost=len(direct_base & removed) / max(len(direct_base), 1),
                        direct_precision=len(kept & pos) / max(len(kept), 1),
                        wrong_share=len((removed & dev.universe) - dev.reach) / max(len(removed & dev.universe), 1))

        rows["no removal"].append(widest(set()))
        rows["50 widest, unchecked"].append(widest(set(top)))
        every = sorted(bits.edges)
        blind_any = [widest(set(rng.sample(every, CHECK_BUDGET))) for _ in range(draws)]
        rows["50 random edges"].append({k: float(np.mean([b[k] for b in blind_any])) for k in blind_any[0]})
        rows["all edges, wrong share"].append(dict(wrong_share=len((bits.edges & dev.universe) - dev.reach) / max(len(bits.edges & dev.universe), 1)))
        for label, answers in (("judge no", judge), ("muse+source no", muse)):
            removed = {e for e, yes in zip(top, answers) if not yes}
            actual = effect(removed)
            blind = [effect(set(rng.sample(top, len(removed)))) for _ in range(draws)]
            labeled = removed & dev.universe
            kept = (set(top) - removed) & dev.universe
            rows[label].append(dict(
                **actual,
                wrong_share_removed=len(labeled - dev.reach) / max(len(labeled), 1),
                wrong_share_kept=len(kept - dev.reach) / max(len(kept), 1),
                blind_false_removed_share=float(np.mean([b["false_removed_share"] for b in blind])),
                blind_true_lost_share=float(np.mean([b["true_lost_share"] for b in blind])),
                blind_closure_f1=float(np.mean([b["closure_f1"] for b in blind])),
                blind_at_least_as_good=float(np.mean([b["closure_f1"] >= actual["closure_f1"] for b in blind])),
            ))
    summary = {label: {**{k: float(np.mean([g[k] for g in group])) for k in group[0]},
                       "false_removed_each_run": [g.get("false_removed_share") for g in group]}
               for label, group in rows.items()}
    log_run("experiments", dict(stage="repair_controls", arm=arm, split="tune", draws=draws, prompt_version=None, git=git_state(), code=code_hash()),
            clean(summary))
    for label, row in summary.items():
        show(label, row)


def precision_of(edges, pos, labeled):
    scored = edges & labeled
    return dict(scored=len(scored), precision=len(scored & pos) / max(len(scored), 1))


def uptake_row(picks, pos, labeled, named):
    """Did picks follow the excerpts? named holds the pairs whose parent is named in the target's real excerpts."""
    gold_named, gold_other = pos & named, pos - named
    return dict(picks=len(picks), share_named=len(picks & named) / max(len(picks), 1),
                named=precision_of(picks & named, pos, labeled), unnamed=precision_of(picks - named, pos, labeled),
                recall_named=len(picks & gold_named) / max(len(gold_named), 1),
                recall_unnamed=len(picks & gold_other) / max(len(gold_other), 1))


def change_row(base, other, pos, labeled):
    """Which edges does a change drop and add? Equal precision for kept and dropped means the pruning is blind."""
    return dict(kept=precision_of(base & other, pos, labeled), dropped=precision_of(base - other, pos, labeled),
                added=precision_of(other - base, pos, labeled))


def mean_rows(rows):
    if isinstance(rows[0], dict):
        return {k: mean_rows([r[k] for r in rows]) for k in rows[0]}
    return float(np.mean(rows))


def uptake():
    """Exploratory: whether the source arm's picks follow what the excerpts say. No model calls. Tune, covered targets."""
    dev = Dev()
    ids = load_split("tune")
    targets = covered_targets(dev, ids)
    library = Library.load()
    text = {t: normal(" ".join(p["text"] for p in library.passages(dev.names[t]))) for t in targets}
    patterns = {i: name_pattern(dev.names[i]) for i in ids}
    named = {(a, t) for a in ids for t in targets if a != t and patterns[a] and patterns[a].search(text[t])}
    in_scope = lambda pairs: {p for p in pairs if p[0] in ids and p[1] in targets}
    pos = in_scope(dev.positive)
    labeled = pos | in_scope(dev.negative)
    load = lambda arm, r: in_scope(set(load_picks(arm, r)))
    reps = range(REPLICATES)
    result = dict(gold_parents=len(pos), gold_parents_named=len(pos & named))
    result["by_arm"] = {arm: mean_rows([uptake_row(load(arm, r), pos, labeled, named) for r in reps])
                        for arm in ("capped", "uncapped", "source", "placebo")}
    result["from_uncapped"] = {arm: mean_rows([change_row(load("uncapped", r), load(arm, r), pos, labeled) for r in reps])
                               for arm in ("source", "placebo")}
    result["from_uncapped"]["another uncapped run"] = mean_rows(
        [change_row(load("uncapped", r), load("uncapped", (r + 1) % REPLICATES), pos, labeled) for r in reps])
    capped = arm_summary(dev, "capped", ids, targets)
    result["capped_on_covered"] = clean({k: capped[k] for k in ("direct", "reach", "raw")})
    log_run("experiments", dict(stage="uptake", split="tune", prompt_version=None, git=git_state(), code=code_hash()), clean(result))
    show("uptake", result)


def dispute_groups(edges):
    """Pairs labeled by both datasets, by outcome: disputed, both yes, both no."""
    labels = defaultdict(dict)
    for e in edges:
        labels[e["prereq_id"], e["dependent_id"]][e["source"]] = e["label"]
    shared = {pair: v for pair, v in labels.items() if len(v) == 2}
    groups = dict(disputed=[], both_yes=[], both_no=[])
    for pair, v in sorted(shared.items()):
        kind = "disputed" if v["lb1"] != v["lb2"] else "both_yes" if v["lb1"] else "both_no"
        groups[kind].append((pair, v))
    return groups


def disputes(per_group=22):
    """Blind sheet for the LB1/LB2 agreement check. Agreed pairs are mixed in so a rater cannot infer the answer."""
    edges = json.loads((SPLIT_DIR / "system_dev.json").read_text())["edges"]
    dev = Dev()
    groups = dispute_groups(edges)
    rng = random.Random(SEED)
    items = groups["disputed"] + groups["both_yes"] + rng.sample(groups["both_no"], per_group)
    rng.shuffle(items)
    with (RUNS / "kappa_review.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["item", "A", "B", "rater_A_needed_for_B"])
        for i, ((a, b), _) in enumerate(items):
            out.writerow([i, dev.names[a], dev.names[b], ""])
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "kappa_key.json").write_text(json.dumps({i: v for i, (_, v) in enumerate(items)}))
    result = dict(items=len(items), **{k: len(v) for k, v in groups.items()})
    log_run("experiments", dict(stage="disputes", split="all dev", seed=SEED, prompt_version=None, git=git_state(), code=code_hash()), result)
    show("disputes", result)


def wilson(hits, n, z=1.96):
    """95% interval for a proportion."""
    if not n:
        return (0.0, 0.0)
    rate = hits / n
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return (centre - half, centre + half)


def tier_of(item, ratings):
    return next((tier for tier in ("yes", "unclear", "no") if item in ratings[tier]), "unrated")


def tally_arm(items, ratings):
    """Precision of one arm's sampled edges under a strict and a lenient reading, plus a floor counting unrated as no."""
    tiers = Counter(tier_of(i, ratings) for i in items)
    rated = tiers["yes"] + tiers["unclear"] + tiers["no"]
    return dict(**{t: tiers[t] for t in ("yes", "unclear", "no", "unrated")}, strict=tiers["yes"] / max(rated, 1),
                strict_interval=wilson(tiers["yes"], rated), lenient=(tiers["yes"] + tiers["unclear"]) / max(rated, 1),
                floor=tiers["yes"] / max(len(items), 1))


def score_reviews():
    """Score written reviews against the keys. The rater comes from the ratings file; AI review is not human review."""
    ratings = json.loads((RUNS / "review_ratings.json").read_text())
    dev = Dev()
    key = json.loads((PRIVATE / "kappa_key.json").read_text())
    kappa = ratings["kappa_review"]
    tier = lambda i: "yes" if i in kappa["yes"] else "defensible" if i in kappa["defensible"] else "no"
    outcome = defaultdict(Counter)
    for item, v in key.items():
        kind = "lb1_only" if v["lb1"] > v["lb2"] else "lb2_only" if v["lb2"] > v["lb1"] else "both_yes" if v["lb1"] else "both_no"
        outcome[kind][tier(int(item))] += 1
    packet = json.loads((PRIVATE / "review_key.json").read_text())
    rows = list(csv.DictReader((RUNS / "review_packet.csv").open()))
    edge = {int(r["item"]): (dev.ids[r["A"]], dev.ids[r["B"]]) for r in rows}
    rated = ratings["review_packet"]
    gold = lambda e: "positive" if e in dev.positive else "negative" if e in dev.negative else "unlabeled"
    crossed = defaultdict(Counter)
    for item, e in edge.items():
        crossed[gold(e)][tier_of(item, rated)] += 1
    arms = {arm: tally_arm(items, rated) for arm, items in packet.items()}
    by_gold = {arm: dict(Counter(gold(edge[i]) for i in items)) for arm, items in packet.items()}
    result = dict(rater=ratings["rater"], kappa_sheet={k: dict(v) for k, v in outcome.items()}, packet_arms=arms,
                  packet_gold_counts=by_gold, packet_gold_by_tier={k: dict(v) for k, v in crossed.items()})
    log_run("experiments", dict(stage="score_reviews", split="all dev", prompt_version=None, git=git_state(), code=code_hash()), clean(result))
    show("reviews", result)


def holdout():
    """One scoring of the frozen choices on concepts no choice has used."""
    if not FREEZE.exists():
        raise FileNotFoundError("write the tune decisions to exp_freeze.json first")
    if LOG.exists() and any(json.loads(line)["config"].get("stage") == "holdout" for line in LOG.open()):
        raise RuntimeError("holdout was already scored")
    frozen = json.loads(FREEZE.read_text())
    dev = Dev()
    ids = load_split("holdout")
    results = {arm: arm_summary(dev, arm, ids) for arm in frozen["arms"]}
    for arm, summary in results.items():
        show(arm, summary)
    gains = {f"{a} vs {b}": {kind: paired_gain(results[a]["counts"][kind], results[b]["counts"][kind]) for kind in ("direct", "reach", "raw")} for a, b in frozen["compare"]}
    for label, gain in gains.items():
        show(label, gain)
    log_run("experiments", dict(stage="holdout", split="holdout", frozen=frozen, prompt_version="see arms", git=git_state(), code=code_hash()),
            clean(dict(arms=results, gains=gains)))


def packet(arms=("capped", "uncapped", "source"), per_arm=40):
    """Blind review sheet: a uniform sample of emitted edges per arm, shuffled, names only. The key stays private."""
    dev = Dev()
    rng = random.Random(SEED)
    sampled = {arm: rng.sample(sorted(load_picks(arm, 0)), per_arm) for arm in arms}
    pooled = sorted({e for edges in sampled.values() for e in edges})
    rng.shuffle(pooled)
    rows = ["item,A,B,rater_1_A_needed_for_B,rater_2_A_needed_for_B"]
    for i, (a, b) in enumerate(pooled):
        rows.append(f'{i},"{dev.names[a]}","{dev.names[b]}",,')
    (RUNS / "review_packet.csv").write_text("\n".join(rows) + "\n")
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "review_key.json").write_text(json.dumps({arm: [pooled.index(e) for e in edges] for arm, edges in sampled.items()}))
    print("items", len(pooled))


def main():
    command, args = sys.argv[1], sys.argv[2:]
    if command == "arm":
        return run_arm(args[0], int(args[1]))
    return dict(split=make_split, report=report, signal=signal, repair=repair, controls=controls, uptake=uptake, votes=votes, holdout=holdout, packet=packet, disputes=disputes, score_reviews=score_reviews)[command](*args)


if __name__ == "__main__":
    main()
