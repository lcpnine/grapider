"""Dev-only: hub label audit, oracle attribution, document-mode shape. No model calls, no test data."""
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from prereq.dag import PrereqDAG
from prereq.pipeline import LEGACY_CAP as MAX_PARENTS
from prereq.llm import CACHE, MUSE_MODEL
from prereq.pipeline import EDGE_PROMPTS, break_cycles, gold_reachability
from prereq.schema import canon
from prereq.splits import SPLIT_DIR

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "data/raw/lecturebank/LB-Paper"
split = json.loads((SPLIT_DIR / "system_dev.json").read_text())
concepts, edges = split["concepts"], split["edges"]
name = {c["id"]: c["name"] for c in concepts}
labels = {}
for e in edges:
    labels.setdefault((e["prereq_id"], e["dependent_id"]), {})[e["source"]] = e["label"]
positive = {p for p, v in labels.items() if 1 in v.values()}
negative = {p for p, v in labels.items() if 1 not in v.values()}
reach, universe = gold_reachability(edges)
gold = nx.DiGraph(positive)

print("== raw loader check, LB1")
topics = list(csv.reader(open(LB / "208topics.csv", encoding="utf8", errors="replace")))
print("first topics", topics[:3], "last", topics[-2:])
t_ids = [r[0] for r in topics if r]
ann = [r for r in csv.reader(open(LB / "prerequisite_annotation.csv", encoding="utf8", errors="replace")) if r]
print("first rows", ann[:3])
ann_ids = Counter(x for r in ann for x in r[:2])
unknown = {k: v for k, v in ann_ids.items() if k not in set(t_ids)}
print("topic ids", len(set(t_ids)), "annotation ids", len(ann_ids), "unknown ids", unknown)
print("labels", Counter(r[2] for r in ann))

print("\n== raw loader check, LB2")
d = LB / "LectureBank2/data"
lines = [l.rstrip("\n") for l in open(d / "322topics_final.tsv", encoding="utf8", errors="replace")]
print("first", lines[:3], "last", lines[-2:], "lines without bar", sum("|" not in l for l in lines))
ann2 = [r for r in csv.reader(open(d / "final_new_annotation.csv", encoding="utf8", errors="replace")) if r]
print("first rows", ann2[:3], "labels", Counter(r[2] for r in ann2))
id2 = dict(l.split("|", 1) for l in lines if "|" in l)
print("annotation ids not in topics", Counter(x for r in ann2 for x in r[:2] if x not in id2).most_common(5))

print("\n== hub parents: where labels come from, and the share of possible targets marked")
hubs = [n for n, deg in sorted(gold.out_degree(), key=lambda x: -x[1])[:8]]
for h in hubs:
    row = Counter()
    for (a, b), v in labels.items():
        if a == h:
            for s, lab in v.items():
                row[s, lab] += 1
    print(f"  {name[h]}: {dict(row)}")
    print("     sample children", [name[b] for a, b in sorted(positive) if a == h][:14])

print("\n== is a hub's child set shared across hubs? (block annotation sign)")
children = {h: {b for a, b in positive if a == h} for h in hubs}
for i, h in enumerate(hubs):
    for k in hubs[i + 1:]:
        u, v = children[h], children[k]
        if len(u & v) / max(min(len(u), len(v)), 1) > 0.5:
            print(f"  {name[h]} & {name[k]}: overlap {len(u & v)} of {len(u)} / {len(v)}")

print("\n== LB1 raw id order of hub labels (runs of positives by id)")
by_id = {r[0]: r[1] for r in topics if r}
for h in hubs[:3]:
    raw = [i for i, n in {**by_id, **id2}.items() if canon(n) == h]
    print(" ", name[h], "raw ids", raw)


def rebuild(prompt_id):
    """Same steps as fixed_concept_dag, from cached answers."""
    dag = PrereqDAG()
    for c in concepts:
        dag.add_node(c["id"], c["name"], [c["source"]])
    ids = {c["name"]: c["id"] for c in concepts}
    cache = {}
    for row in map(json.loads, CACHE.open()):
        cache.setdefault(row["key"], row)
    for target in ids:
        others = sorted(n for n in ids if n != target)
        user = f'Target: "{target}"\nConcepts:\n' + "\n".join(f"- {n}" for n in others)
        key = hashlib.sha256(json.dumps([MUSE_MODEL, EDGE_PROMPTS[prompt_id], user]).encode()).hexdigest()
        known = {canon(n): n for n in others}
        chosen = []
        for p in cache[key]["out"].get("prereqs", []):
            n = known.get(canon(str(p.get("name", ""))))
            if n and n not in [c[0] for c in chosen]:
                chosen.append((n, "x"))
        for n, why in chosen[:MAX_PARENTS]:
            dag.add_edge(ids[n], ids[target], why)
    raw = set(dag.g.edges)
    removed = break_cycles(dag)
    dag.transitive_reduce()
    return dag, raw, removed


def score(edge_set, positives=reach, pool=universe):
    g = nx.DiGraph(edge_set)
    if not nx.is_directed_acyclic_graph(g):
        tmp = PrereqDAG()
        for n in g:
            tmp.add_node(n, n, ["x"])
        for a, b in edge_set:
            tmp.add_edge(a, b, "x")
        break_cycles(tmp)
        g = tmp.g
    closure = {(u, v) for u, v in nx.transitive_closure(g, reflexive=False).edges if u != v}
    in_scope = closure & pool
    tp = len(in_scope & positives)
    p, r = tp / max(len(in_scope), 1), tp / max(len(positives), 1)
    return dict(P=round(p, 3), R=round(r, 3), F1=round(2 * p * r / (p + r), 3) if tp else 0, predicted=len(in_scope))


dag, raw, removed = rebuild("e2_strict")
final = set(dag.g.edges)
print("\n== e2_strict rebuilt from cache")
print("raw", len(raw), "cycle removed", len(removed), "final", len(final), "baseline", score(final))
print("cycle-broken edges that were gold direct or in closure", sum(e in reach for e in removed), "of", len(removed))
reduced_away = raw - final - set(removed)
print("reduced away", len(reduced_away), "of them gold direct", sum(e in positive for e in reduced_away))

print("\n== oracle attribution: fix one failure class with gold, keep the rest")
hub7 = {n for n, deg in gold.out_degree() if deg >= 26}
print("hubs (>=26 children)", [name[h] for h in hub7], "their positives", sum(a in hub7 for a, _ in positive), "of", len(positive))
closure = {(u, v) for u, v in nx.transitive_closure(dag.g, reflexive=False).edges if u != v}
hub_reach = {p for p in reach if p[0] in hub7}
print(f"closure recall, hub-parent pairs {len(closure & hub_reach) / len(hub_reach):.3f} ({len(hub_reach)}), "
      f"other pairs {len(closure & (reach - hub_reach)) / len(reach - hub_reach):.3f} ({len(reach - hub_reach)})")
wrong_final = {e for e in final if e not in reach and e in universe}
reversed_final = {e for e in final if e[::-1] in reach and e not in reach}
rows = {
    "baseline": final,
    "remove reversed edges": final - reversed_final,
    "flip reversed edges": (final - reversed_final) | {e[::-1] for e in reversed_final},
    "remove all labeled-wrong edges": final - wrong_final,
    "remove labeled-wrong and unlabeled edges": final & reach,
    "add gold hub-parent edges": final | {p for p in positive if p[0] in hub7},
    "add gold non-hub edges": final | {p for p in positive if p[0] not in hub7},
    "add all gold direct edges": final | positive,
    "gold direct only": set(positive),
}
for label, e in rows.items():
    print(f"  {label:42} {score(e)}")

print("\n== same model output, score against a gold with the 7 hub parents removed")
g2 = nx.DiGraph(p for p in positive if p[0] not in hub7)
reach2 = {(u, v) for u, v in nx.transitive_closure(g2, reflexive=False).edges if u != v}
pool2 = (universe - {p for p in universe if p[0] in hub7}) | reach2
print("  gold closure without hubs", len(reach2), "prevalence", round(len(reach2) / len(pool2), 4))
print("  e2_strict", score({e for e in final if e[0] not in hub7}, reach2, pool2))

print("\n== parent-cap oracle: gold direct edges, only 3 parents per target, closure scored against full gold")
rng = random.Random(0)
for cap in (1, 2, 3, 5, 8, 100):
    runs = []
    for seed in range(5):
        rng = random.Random(seed)
        kept = set()
        for t in gold:
            ps = sorted(gold.predecessors(t))
            rng.shuffle(ps)
            kept |= {(p, t) for p in ps[:cap]}
        runs.append(score(kept))
    print(f"  cap {cap}: recall {sum(r['R'] for r in runs) / 5:.3f}, F1 {sum(r['F1'] for r in runs) / 5:.3f}")

print("\n== closure error amplification, e2_strict")
tree = nx.DiGraph(final)
fp = {p for p in closure & universe if p not in reach}
print("closure scored", len(closure & universe), "false", len(fp), "from direct edges", len(fp & final))
contrib = Counter()
order = sorted(final)
for e in order:
    if e in reach:
        continue
    g = dag.g.copy()
    g.remove_edge(*e)
    c = {(u, v) for u, v in nx.transitive_closure(g, reflexive=False).edges if u != v}
    contrib[e] = len(fp - c)
print("wrong final edges", len(contrib), "edges with largest effect", [(name[a], name[b], n) for (a, b), n in contrib.most_common(8)])
print("share of false closure pairs touched by top 10 wrong edges", round(sum(n for _, n in contrib.most_common(10)) / max(len(fp), 1), 3))

print("\n== document mode shape, Civics dev")
for f in ("civics_2008", "civics_2025_rebuild", "civics_2025_rebuild_reversed"):
    data = json.loads((ROOT / "runs/graphs" / f"{f}.json").read_text())
    nodes = data["nodes"]
    active = [n for n in nodes if n["provenance"]]
    sizes = Counter(len(n["provenance"]) for n in active)
    qs = {q for n in active for q in n["provenance"]}
    print(f, "nodes", len(active), "questions", len(qs), "edges", len(data["edges"]),
          "single-question nodes", sizes[1], f"({sizes[1] / len(active):.2f})")
data = json.loads((ROOT / "runs/graphs/civics_2008.json").read_text())
names = [n["name"] for n in data["nodes"]]
stop = {"the", "of", "a", "an", "and", "in", "to", "for", "us", "u", "s", "united", "states", "american", "america"}
toks = {n: set(re.findall(r"[a-z0-9]+", n.lower())) - stop for n in names}
near = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if toks[a] and toks[a] == toks[b]]
sub = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if toks[a] and toks[b] and (toks[a] < toks[b] or toks[b] < toks[a])]
print("same key words", len(near), near[:15])
print("one name's words inside another", len(sub), rng.sample(sub, min(20, len(sub))))
civ = dict(a=set(map(tuple, [(e["prereq"], e["dependent"]) if "prereq" in e else (e["source"], e["target"]) for e in json.loads((ROOT / "runs/graphs/civics_2025_rebuild.json").read_text())["edges"]])),
           b=set(map(tuple, [(e["prereq"], e["dependent"]) if "prereq" in e else (e["source"], e["target"]) for e in json.loads((ROOT / "runs/graphs/civics_2025_rebuild_reversed.json").read_text())["edges"]])))
print("civics rebuild vs reversed list: edges", len(civ["a"]), len(civ["b"]), "shared", len(civ["a"] & civ["b"]),
      "jaccard", round(len(civ["a"] & civ["b"]) / len(civ["a"] | civ["b"]), 3))
