"""Revision agent: a deterministic workflow, with an LLM judgment step only on anomalies.

Usage: python -m prereq.revision
"""
import json
import random

import networkx as nx

from .civics import diff_questions, load_civics
from .dag import PrereqDAG
from .judge import FROZEN, predict
from .llm import MUSE_MODEL, Ledger, ask_json, log_run
from .pipeline import DEFAULT_EDGE_PROMPT, add_questions, break_cycles, build_edges, build_from_questions
from .schema import ROOT

OUT = ROOT / "runs" / "graphs"
LEARNER_PROGRESS = 0.4
CYCLE_SYSTEM = ('These prerequisite edges form a cycle, so one must be removed. Pick the edge least likely to be a '
                'true prerequisite. Reply with JSON only: {"prereq": "<id>", "dependent": "<id>"}.')


def cycle_judge(ledger, anomalies):
    def judge(cycle):
        anomalies["cycle"] += 1
        lines = "\n".join(f"{u} -> {v}" for u, v in cycle)
        out = ask_json(CYCLE_SYSTEM, lines, ledger)
        return (out.get("prereq"), out.get("dependent"))
    return judge


def retire_node(dag, node, anomalies):
    """Remove a node and reconnect its parents to its children, so paths survive."""
    parents, children = list(dag.g.predecessors(node)), list(dag.g.successors(node))
    if parents and children:
        anomalies["broken_path_bridged"] += 1
    for p in parents:
        for c in children:
            if p != c and not dag.g.has_edge(p, c):
                dag.add_edge(p, c, f"bridged after retiring {node}")
    dag.g.remove_node(node)


def enforce_parent_limit(dag):
    """Keep the bridged graph inside the parent limit by dropping bridge edges first."""
    for node in [n for n in dag.g if dag.g.in_degree(n) > 3]:
        parents = sorted(dag.g.predecessors(node), key=lambda p: (not dag.g[p][node]["reason"].startswith("bridged"), p))
        for p in parents[: dag.g.in_degree(node) - 3]:
            dag.g.remove_edge(p, node)


def revise(old_dag, diff, new_questions, version, ledger):
    """Apply a question diff: only nodes the diff touches change. Returns (dag, report)."""
    dag = PrereqDAG.from_json(old_dag.to_json())
    anomalies = dict(cycle=0, broken_path_bridged=0)
    by_id = {q["id"]: q for q in new_questions}
    gone = set(diff["removed"]) | {a for a, _ in diff["modified"]}
    renamed = dict(diff["unchanged"])
    affected, retired = set(), []
    for node, attrs in list(dag.g.nodes(data=True)):
        before = set(attrs["provenance"])
        if before & gone:
            affected.add(node)
        attrs["provenance"] = {renamed.get(p, p) for p in before - gone}
        if attrs["provenance"]:
            attrs["valid_versions"].add(version)
    for node in [n for n, a in dag.g.nodes(data=True) if not a["provenance"]]:
        retired.append(node)
        retire_node(dag, node, anomalies)
    before_nodes = set(dag.g)
    incoming = [by_id[b] for _, b in diff["modified"]] + [by_id[b] for b in diff["added"]]
    affected |= add_questions(dag, incoming, version, ledger)
    new_nodes = sorted(set(dag.g) - before_nodes)
    judge = cycle_judge(ledger, anomalies)
    build_edges(dag, DEFAULT_EDGE_PROMPT, ledger,
                targets=[dag.g.nodes[n]["name"] for n in new_nodes], judge_cycle=judge)
    enforce_parent_limit(dag)
    return dag, dict(affected=sorted(affected), retired=retired, new_nodes=new_nodes, anomalies=anomalies)


def append_only(old_dag, diff, new_questions, version, ledger):
    """Baseline: add concepts from new and modified questions, and never remove anything."""
    dag = PrereqDAG.from_json(old_dag.to_json())
    by_id = {q["id"]: q for q in new_questions}
    renamed = dict(diff["unchanged"])
    for _, attrs in dag.g.nodes(data=True):
        attrs["provenance"] = {renamed.get(p, p) for p in attrs["provenance"]}
    before = set(dag.g)
    incoming = [by_id[b] for _, b in diff["modified"]] + [by_id[b] for b in diff["added"]]
    touched = add_questions(dag, incoming, version, ledger)
    new_nodes = sorted(set(dag.g) - before)
    build_edges(dag, DEFAULT_EDGE_PROMPT, ledger, targets=[dag.g.nodes[n]["name"] for n in new_nodes])
    return dag, dict(affected=sorted(touched))


def gold_affected(old_dag, diff, new_dag):
    """Nodes whose provenance includes an added, removed or modified question."""
    old_changed = set(diff["removed"]) | {a for a, _ in diff["modified"]}
    new_changed = set(diff["added"]) | {b for _, b in diff["modified"]}
    gold = {n for n, a in old_dag.g.nodes(data=True) if a["provenance"] & old_changed}
    gold |= {n for n, a in new_dag.g.nodes(data=True) if a["provenance"] & new_changed}
    return gold


def prf(predicted, gold):
    tp = len(predicted & gold)
    p, r = tp / len(predicted) if predicted else 0.0, tp / len(gold) if gold else 0.0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if tp else 0.0)


def study_order(dag):
    return list(nx.lexicographical_topological_sort(dag.g))


def plan_violations(plan, known, reference):
    """Count concepts studied before a prerequisite that the reference graph says comes first.

    A reference prerequisite the plan never teaches also counts, unless already known.
    """
    position = {n: i for i, n in enumerate(plan)}
    count = 0
    for node in plan:
        if node not in reference:
            continue
        for ancestor in nx.ancestors(reference, node):
            if ancestor not in known and position.get(ancestor, len(plan)) > position[node]:
                count += 1
    return count


def learner_report(old_dag, new_dag, reference):
    """A learner is 40% through the old plan when the revision lands."""
    old_plan = study_order(old_dag)
    known = set(old_plan[: int(len(old_plan) * LEARNER_PROGRESS)])
    remaining = [n for n in study_order(new_dag) if n not in known]
    current = {n for n, a in reference.nodes(data=True) if a["provenance"]}
    return dict(
        known=len(known), remaining=len(remaining),
        violations_vs_reference=plan_violations(remaining, known, reference),
        violations_own_graph=plan_violations(remaining, known, new_dag.g),
        stale_in_remaining=sum(1 for n in remaining if not _current(new_dag, n)),
        reference_concepts_missing=len(current - set(new_dag.g) - known),
    )


def _current(dag, node):
    """True if a current-version question still supports this node."""
    return any(p.startswith("2025:") for p in dag.g.nodes[node]["provenance"])


def judge_precision(dag, sample=150, seed=0):
    """Share of sampled edges the frozen judge calls true. Reported, never tuned on."""
    frozen = json.loads(FROZEN.read_text())
    edges = sorted(dag.g.edges)
    random.Random(seed).shuffle(edges)
    pairs = [(dag.g.nodes[u]["name"], dag.g.nodes[v]["name"]) for u, v in edges[:sample]]
    votes = predict(frozen["model"], frozen["prompt_version"], pairs)
    return sum(votes) / len(votes) if votes else 0.0


def evaluate(name, dag, report, ledger, old_dag, diff, reference):
    gold = gold_affected(old_dag, diff, reference)
    predicted = set(report["affected"])
    kept_unaffected = [e for e in old_dag.g.edges
                       if e[0] not in gold and e[1] not in gold]
    retained = sum(dag.g.has_edge(*e) for e in kept_unaffected)
    return dict(
        method=name, nodes=len(dag.g), edges=len(dag.g.edges),
        affected=prf(predicted, gold), gold_affected=len(gold), predicted_affected=len(predicted),
        unaffected_edges_retained=retained / len(kept_unaffected) if kept_unaffected else 1.0,
        stale_nodes=sum(1 for n in dag.g if not _current(dag, n)),
        validation={k: len(v) for k, v in dag.validate().items()},
        learner=learner_report(old_dag, dag, reference.g),
        judge_edge_precision=judge_precision(dag),
        anomalies=report.get("anomalies"), **ledger.to_dict(),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_q, new_q = load_civics()
    assert (len(old_q), len(new_q)) == (100, 128), (len(old_q), len(new_q))
    diff = diff_questions(old_q, new_q)
    print("diff", {k: len(v) for k, v in diff.items()}, flush=True)
    (OUT / "civics_diff.json").write_text(json.dumps(diff, indent=1))

    base_ledger = Ledger()
    old_dag, old_report = build_from_questions(old_q, "2008", ledger=base_ledger)
    old_dag.save(OUT / "civics_2008.json")
    print("2008", len(old_dag.g), len(old_dag.g.edges), old_report, base_ledger.to_dict(), flush=True)

    rebuild_ledger = Ledger()
    rebuilt, rebuild_report = build_from_questions(new_q, "2025", ledger=rebuild_ledger)
    rebuilt.save(OUT / "civics_2025_rebuild.json")
    gold = gold_affected(old_dag, diff, rebuilt)
    changed = (set(old_dag.g) ^ set(rebuilt.g)) | {n for e in set(old_dag.g.edges) ^ set(rebuilt.g.edges) for n in e}
    rows = [evaluate("full_rebuild", rebuilt, dict(affected=sorted(changed), anomalies=None), rebuild_ledger,
                     old_dag, diff, rebuilt)]

    ledger = Ledger()
    revised, report = revise(old_dag, diff, new_q, "2025", ledger)
    revised.save(OUT / "civics_2025_revised.json")
    rows.append(evaluate("revision_agent", revised, report, ledger, old_dag, diff, rebuilt))

    ledger = Ledger()
    appended, report = append_only(old_dag, diff, new_q, "2025", ledger)
    appended.save(OUT / "civics_2025_append.json")
    rows.append(evaluate("append_only", appended, report, ledger, old_dag, diff, rebuilt))

    for row in rows:
        log_run("revision", dict(stage="civics_dev", method=row["method"], prompt_version=DEFAULT_EDGE_PROMPT, model=MUSE_MODEL), row)
        print(json.dumps(row), flush=True)
    print("gold affected", len(gold))


if __name__ == "__main__":
    main()
