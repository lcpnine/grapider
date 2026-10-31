"""Dev-only failure analysis from cached answers. Calls no model and reads no test data."""
import hashlib
import json
import random
import statistics as st
import sys
from collections import Counter

import networkx as nx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from prereq.dag import MAX_PARENTS
from prereq.llm import CACHE, MUSE_MODEL
from prereq.pipeline import EDGE_PROMPTS, gold_reachability
from prereq.schema import canon
from prereq.splits import SPLIT_DIR

split = json.loads((SPLIT_DIR / "system_dev.json").read_text())
concepts, edges = split["concepts"], split["edges"]
name = {c["id"]: c["name"] for c in concepts}
ids = {n: i for i, n in name.items()}
cache = {}
all_rows = Counter()
answers = {}
for row in map(json.loads, CACHE.open()):
    all_rows[row["key"]] += 1
    cache.setdefault(row["key"], row)
    answers.setdefault(row["key"], []).append(row["out"])


def cached_picks(prompt_id, target):
    others = sorted(n for n in name.values() if n != target)
    user = f'Target: "{target}"\nConcepts:\n' + "\n".join(f"- {n}" for n in others)
    key = hashlib.sha256(json.dumps([MUSE_MODEL, EDGE_PROMPTS[prompt_id], user]).encode()).hexdigest()
    return key, others


print("== data")
print("concepts", len(concepts), "MAX_PARENTS", MAX_PARENTS, "labeled rows", len(edges))
by_src = Counter((e["source"], e["label"]) for e in edges)
print("rows by source/label", dict(by_src))

# Labels by pair and dataset
labels = {}
for e in edges:
    labels.setdefault((e["prereq_id"], e["dependent_id"]), {})[e["source"]] = e["label"]
both = {p: v for p, v in labels.items() if len(v) == 2}
pos1 = {p for p, v in both.items() if v["lb1"] == 1}
pos2 = {p for p, v in both.items() if v["lb2"] == 1}
print("pairs labeled by both", len(both), "lb1 pos", len(pos1), "lb2 pos", len(pos2), "both pos", len(pos1 & pos2),
      "union", len(pos1 | pos2))
if pos1 or pos2:
    tp = len(pos1 & pos2)
    p, r = tp / len(pos2) if pos2 else 0, tp / len(pos1) if pos1 else 0
    print(f"  agreement: lb2 predicting lb1 P {p:.3f} R {r:.3f} F1 {2 * p * r / (p + r) if tp else 0:.3f}")
    n = len(both)
    a = sum(v["lb1"] == v["lb2"] for v in both.values()) / n
    pe = (len(pos1) / n) * (len(pos2) / n) + (1 - len(pos1) / n) * (1 - len(pos2) / n)
    print(f"  raw agreement {a:.3f}, kappa {(a - pe) / (1 - pe):.3f}")

positive = {p for p, v in labels.items() if 1 in v.values()}
negative = {p for p, v in labels.items() if 1 not in v.values()}
reach, universe = gold_reachability(edges)
gold = nx.DiGraph(positive)
sccs = [c for c in nx.strongly_connected_components(gold) if len(c) > 1]
print("direct positives", len(positive), "negatives", len(negative), "closure", len(reach), "universe", len(universe))
print("gold DAG", nx.is_directed_acyclic_graph(gold), "cyclic groups", len(sccs), "largest", max(map(len, sccs), default=0))
print("labeled-0 pairs closure calls positive", len(reach & negative))
print("reverse pairs both positive after merge", sum((b, a) in positive for a, b in positive) // 2)
out_d = sorted((d for _, d in gold.out_degree()), reverse=True)
in_d = sorted((d for _, d in gold.in_degree()), reverse=True)
print("gold in-degree: max", in_d[0], "median", st.median(in_d), "targets with any parent", len(in_d),
      "of", len(concepts), "over cap", sum(d > MAX_PARENTS for d in in_d))
print("gold out-degree top 10", out_d[:10], "share of positives from top 10 parents", round(sum(out_d[:10]) / len(positive), 3))
print("top gold parents", [(name.get(n, n), d) for n, d in sorted(gold.out_degree(), key=lambda x: -x[1])[:10]])
print("top gold targets", [(name.get(n, n), d) for n, d in sorted(gold.in_degree(), key=lambda x: -x[1])[:10]])

# Ceiling from the parent cap: keep the top MAX_PARENTS gold parents per target
capped = {(p, t) for t in gold for p in list(gold.predecessors(t))[:MAX_PARENTS]}
print("direct recall ceiling under cap", round(len(capped) / len(positive), 3))
reduced_edges = None
if nx.is_directed_acyclic_graph(gold):
    reduced = nx.transitive_reduction(gold)
    rd = [d for _, d in reduced.in_degree()]
    print("gold reduced edges", reduced.number_of_edges(), "over cap after reduction", sum(d > MAX_PARENTS for d in rd))

print("\n== raw picks by prompt (before cycle breaking and reduction)")
results = {}
for prompt_id in EDGE_PROMPTS:
    picks, sizes, dropped, missing, positions = [], [], 0, 0, []
    for target in name.values():
        key, others = cached_picks(prompt_id, target)
        if key not in cache:
            missing += 1
            continue
        known = {canon(n): n for n in others}
        chosen = []
        for p in cache[key]["out"].get("prereqs", []):
            n = known.get(canon(str(p.get("name", ""))))
            if n and n not in chosen:
                chosen.append(n)
            elif not n:
                dropped += 1
        chosen = chosen[:MAX_PARENTS]
        sizes.append(len(chosen))
        for n in chosen:
            picks.append((ids[n], ids[target]))
            positions.append(others.index(n) / len(others))
    picks = set(picks)
    tp = picks & positive
    rev = {(a, b) for a, b in picks if (b, a) in positive}
    rev_reach = {(a, b) for a, b in picks if (b, a) in reach and (a, b) not in reach}
    indirect = (picks & reach) - positive
    labeled_neg_other = (picks & negative) - reach - rev_reach
    unlabeled = picks - positive - negative
    unlabeled_outside = unlabeled - reach
    print(f"{prompt_id}: missing cache {missing}, picks {len(picks)}, mean per target {st.mean(sizes):.2f}, "
          f"at cap {sum(s == MAX_PARENTS for s in sizes)}, empty {sum(s == 0 for s in sizes)}, unknown names {dropped}")
    print(f"   direct gold hit {len(tp)} ({len(tp) / len(picks):.3f}); indirect but in gold closure {len(indirect)} "
          f"({len(indirect) / len(picks):.3f}); reversed direction of closure {len(rev_reach)} ({len(rev_reach) / len(picks):.3f}); "
          f"labeled 0 and unrelated in closure {len(labeled_neg_other)} ({len(labeled_neg_other) / len(picks):.3f}); "
          f"unlabeled {len(unlabeled_outside)} ({len(unlabeled_outside) / len(picks):.3f})")
    print(f"   direct recall {len(tp) / len(positive):.3f}; picks in closure / closure {len(picks & reach) / len(reach):.3f}; "
          f"flipped direct hits {len({(b, a) for a, b in picks} & positive)}")
    print(f"   mean list position of picks {st.mean(positions):.3f} (0.5 = no alphabetical bias)")
    deciles = Counter(int(x * 10) for x in positions)
    print("   picks by list decile", [deciles[i] for i in range(10)])
    if prompt_id == "e2_strict":
        results = dict(picks=picks)

picks = results["picks"]

print("\n== e2_strict: recall by list position of the gold parent")
others_for = {t: sorted(n for n in name.values() if n != t) for t in name.values()}
bins_all, bins_hit = Counter(), Counter()
for a, b in positive:
    if a in name and b in name:
        pos = others_for[name[b]].index(name[a]) / len(others_for[name[b]])
        d = int(pos * 5)
        bins_all[d] += 1
        bins_hit[d] += (a, b) in picks
print([f"{bins_hit[i]}/{bins_all[i]}={bins_hit[i] / max(bins_all[i], 1):.3f}" for i in range(5)])

print("\n== e2_strict: recall by gold in-degree of the target, and by gold out-degree of the parent")
for label, degree, idx in (("target in-degree", dict(gold.in_degree()), 1), ("parent out-degree", dict(gold.out_degree()), 0)):
    groups = {"1-2": (1, 2), "3-5": (3, 5), "6-10": (6, 10), "11-25": (11, 25), "26+": (26, 10_000)}
    row = []
    for g, (lo, hi) in groups.items():
        pool = [p for p in positive if lo <= degree[p[idx]] <= hi]
        row.append(f"{g}: {sum(p in picks for p in pool)}/{len(pool)}={sum(p in picks for p in pool) / max(len(pool), 1):.3f}")
    print(label, row)

print("\n== per-source scoring of raw picks, direct and via closure of raw picks")
g_pick = nx.DiGraph(picks)
for s in ("lb1", "lb2"):
    pos_s = {p for p, v in labels.items() if v.get(s) == 1}
    neg_s = {p for p, v in labels.items() if v.get(s) == 0}
    in_s = picks & (pos_s | neg_s)
    tp = len(in_s & pos_s)
    print(s, f"pos {len(pos_s)} neg {len(neg_s)} prevalence {len(pos_s) / (len(pos_s) + len(neg_s)):.4f}; "
          f"scored picks {len(in_s)}, P {tp / max(len(in_s), 1):.3f} R {tp / max(len(pos_s), 1):.3f}")

print("\n== name ambiguity in the dev candidate list")
words = Counter()
for n in name.values():
    words[" ".join(n.lower().split())] += 1
short = [n for n in name.values() if len(n) <= 4]
print("names of <=4 chars", len(short), short[:40])

print("\n== replicate answers in the cache")
dups = {k: v for k, v in answers.items() if len(v) > 1}
print("keys answered more than once", len(dups), "with different answers", sum(any(x != v[0] for x in v) for v in dups.values()))

rng = random.Random(0)
fp = sorted(p for p in picks if p in negative and p not in reach)
miss = sorted(p for p in positive if p not in picks and p in reach)
print("\n== sample of 40 picks labeled 0 and outside closure")
for a, b in rng.sample(fp, min(40, len(fp))):
    print(f"  {name[a]}  ->  {name[b]}")
print("\n== sample of 40 missed gold direct positives")
for a, b in rng.sample(miss, min(40, len(miss))):
    print(f"  {name.get(a, a)}  ->  {name.get(b, b)}")
