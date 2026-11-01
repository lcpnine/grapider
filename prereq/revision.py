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
from .pipeline import DEFAULT_EDGE_PROMPT, MERGE_SYSTEM, add_questions, build_edges, build_from_questions
from .schema import ROOT

OUT = ROOT / "runs" / "graphs"
LEARNER_PROGRESS = 0.4
RANDOM_ORDER_ROUNDS = 50
CYCLE_SYSTEM = ('These prerequisite edges form a cycle, so one must be removed. Pick the edge least likely to be a '
                'true prerequisite. Reply with JSON only: {"prereq": "<id>", "dependent": "<id>"}.')
CONFLICT_SYSTEM = ('A prerequisite graph has an edge, and new study material suggests the opposite direction. Decide '
                   'which direction a learner should follow. Reply with JSON only: '
                   '{"keep": "current" or "proposed"}.')


def cycle_judge(ledger, anomalies):
    def judge(cycle):
        anomalies["cycle"] += 1
        lines = "\n".join(f"{u} -> {v}" for u, v in cycle)
        out = ask_json(CYCLE_SYSTEM, lines, ledger)
        return (out.get("prereq"), out.get("dependent"))
    return judge


def conflict_judge(dag, ledger, anomalies):
    """Decide a proposal that reverses an existing edge. Anything unclear keeps the current edge."""
    def judge(prereq, dependent, reason):
        anomalies["conflict"] += 1
        a, b = dag.g.nodes[prereq]["name"], dag.g.nodes[dependent]["name"]
        user = (f'Current: "{b}" before "{a}": {dag.g[dependent][prereq]["reason"]}\n'
                f'Proposed: "{a}" before "{b}": {reason}')
        return ask_json(CONFLICT_SYSTEM, user, ledger).get("keep") == "proposed"
    return judge


def absorb_renames(dag, diff, before, new_nodes, ledger, anomalies):
    """A modified question can name the same concept differently. Keep the old node and its edges.

    Returns {absorbed_new_node: old_node}.
    """
    absorbed = {}
    for a, b in diff["modified"]:
        orphaned = [n for n in sorted(before) if a in before[n] and not dag.g.nodes[n]["provenance"]]
        for old in orphaned:
            for new in sorted(n for n in new_nodes - set(absorbed) if b in dag.g.nodes[n]["provenance"]):
                anomalies["ambiguous_merge"] += 1
                user = f'A: "{dag.g.nodes[old]["name"]}"\nB: "{dag.g.nodes[new]["name"]}"'
                if ask_json(MERGE_SYSTEM, user, ledger).get("same"):
                    attrs = dag.g.nodes[new]
                    dag.add_node(old, provenance=attrs["provenance"], valid_versions=attrs["valid_versions"])
                    dag.g.remove_node(new)
                    absorbed[new] = old
                    break
    return absorbed


def retire_node(dag, node, anomalies, version):
    """Retire a node and reconnect its parents to its children, so paths survive."""
    parents, children = list(dag.g.predecessors(node)), list(dag.g.successors(node))
    if parents and children:
        anomalies["broken_path_bridged"] += 1
    for p in parents:
        for c in children:
            if p != c and not dag.g.has_edge(p, c):
                dag.add_edge(p, c, f"bridged after retiring {node}")
    dag.retire(node, version)


def revise(old_dag, diff, new_questions, version, ledger):
    """Apply a question diff: only nodes the diff touches change. Returns (dag, report)."""
    dag = PrereqDAG.from_json(old_dag.to_json())
    anomalies = dict(cycle=0, broken_path_bridged=0, conflict=0, ambiguous_merge=0)
    by_id = {q["id"]: q for q in new_questions}
    gone = set(diff["removed"]) | {a for a, _ in diff["modified"]}
    renamed = dict(diff["unchanged"])
    affected, before = set(), {}
    for node, attrs in dag.g.nodes(data=True):
        before[node] = set(attrs["provenance"])
        if before[node] & gone:
            affected.add(node)
        attrs["provenance"] = {renamed.get(p, p) for p in before[node] - gone}
        if attrs["provenance"]:
            attrs["valid_versions"].add(version)

    # Read the new text before retiring anything, so a concept that survives keeps its edges.
    incoming = [by_id[b] for _, b in diff["modified"]] + [by_id[b] for b in diff["added"]]
    touched = add_questions(dag, incoming, version, ledger, anomalies)
    new_nodes = set(dag.g) - set(before)
    absorbed = absorb_renames(dag, diff, before, new_nodes, ledger, anomalies)
    new_nodes -= set(absorbed)
    touched = (touched - set(absorbed)) | set(absorbed.values())

    retired = sorted(n for n in before if not dag.g.nodes[n]["provenance"])
    for node in retired:
        dag.g.nodes[node]["provenance"] = before[node]
        retire_node(dag, node, anomalies, version)

    # New nodes get parents. An existing node a new question touched can only gain a new concept
    # as a parent, unless its proposal reverses an edge, which is a conflict for the judge to decide.
    build_edges(dag, DEFAULT_EDGE_PROMPT, ledger,
                targets=[dag.g.nodes[n]["name"] for n in sorted(new_nodes | touched)],
                judge_cycle=cycle_judge(ledger, anomalies),
                keep=lambda prereq, dependent: prereq in new_nodes or dependent in new_nodes,
                on_conflict=conflict_judge(dag, ledger, anomalies))
    return dag, dict(affected=sorted(affected | touched), retired=retired, new_nodes=sorted(new_nodes),
                     absorbed=absorbed, anomalies=anomalies)


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


def violated_pairs(plan, known, reference):
    """(prereq, concept) pairs where a concept is studied before what the reference says comes first.

    A reference prerequisite the plan never teaches also counts, unless already known.
    """
    position = {n: i for i, n in enumerate(plan)}
    pairs = []
    for node in plan:
        if node not in reference:
            continue
        for ancestor in sorted(nx.ancestors(reference, node)):
            if ancestor not in known and position.get(ancestor, len(plan)) > position[node]:
                pairs.append((ancestor, node))
    return pairs


def plan_violations(plan, known, reference):
    return len(violated_pairs(plan, known, reference))


def plan_orderings(plan, known, reference):
    """How many prerequisite orderings the plan could violate: the denominator for plan_violations."""
    return sum(len(nx.ancestors(reference, n) - known) for n in plan if n in reference)


def random_order_violations(plan, known, reference):
    """Mean violations of the same concepts in random order, as a floor for what the count means."""
    total = 0
    for seed in range(RANDOM_ORDER_ROUNDS):
        shuffled = list(plan)
        random.Random(seed).shuffle(shuffled)
        total += plan_violations(shuffled, known, reference)
    return total / RANDOM_ORDER_ROUNDS


def learner_position(old_dag, new_dag):
    """A learner is 40% through the old plan when the revision lands: what they know and have left."""
    old_plan = study_order(old_dag)
    known = set(old_plan[: int(len(old_plan) * LEARNER_PROGRESS)])
    return known, [n for n in study_order(new_dag) if n not in known]


def learner_report(old_dag, new_dag, reference, version):
    known, remaining = learner_position(old_dag, new_dag)
    current = {n for n, a in reference.nodes(data=True) if a["provenance"]}
    violations, orderings = plan_violations(remaining, known, reference), plan_orderings(remaining, known, reference)
    return dict(
        known=len(known), remaining=len(remaining),
        violations_vs_reference=violations, reference_orderings=orderings,
        violation_rate=violations / orderings if orderings else 0.0,
        random_order_violations=random_order_violations(remaining, known, reference),
        violations_own_graph=plan_violations(remaining, known, new_dag.g),
        stale_in_remaining=sum(1 for n in remaining if not _current(new_dag, n, version)),
        reference_concepts_missing=len(current - set(new_dag.g) - known),
    )


def _current(dag, node, version):
    """True if a current-version question still supports this node."""
    return any(p.startswith(f"{version}:") for p in dag.g.nodes[node]["provenance"])


def judge_precision(dag, sample=150, seed=0):
    """Share of sampled edges the frozen judge calls true. Reported, never tuned on."""
    frozen = json.loads(FROZEN.read_text())
    edges = sorted(dag.g.edges)
    random.Random(seed).shuffle(edges)
    pairs = [(dag.g.nodes[u]["name"], dag.g.nodes[v]["name"]) for u, v in edges[:sample]]
    votes = predict(frozen["model"], frozen["prompt_version"], pairs)
    return sum(votes) / len(votes) if votes else 0.0


def judge_confirmed_violations(old_dag, dag, reference):
    """Violated pairs the frozen judge also calls true prerequisites, split by what the plan did.

    The judge has high precision and low recall on dev, so this is a floor on real violations.
    """
    known, remaining = learner_position(old_dag, dag)
    pairs = violated_pairs(remaining, known, reference.g)
    if not pairs:
        return dict(judge_confirmed=0, prereq_never_taught=0, judge_confirmed_out_of_order=0)
    frozen = json.loads(FROZEN.read_text())
    names = reference.g.nodes
    votes = predict(frozen["model"], frozen["prompt_version"], [(names[a]["name"], names[b]["name"]) for a, b in pairs])
    taught = set(remaining)
    return dict(judge_confirmed=sum(votes), prereq_never_taught=sum(a not in taught for a, _ in pairs),
                judge_confirmed_out_of_order=sum(v for v, (a, _) in zip(votes, pairs) if a in taught))


def evaluate(name, dag, report, ledger, old_dag, diff, reference, version):
    gold = gold_affected(old_dag, diff, reference)
    predicted = set(report["affected"])
    kept_unaffected = [e for e in old_dag.g.edges
                       if e[0] not in gold and e[1] not in gold]
    retained = sum(dag.g.has_edge(*e) for e in kept_unaffected)
    closure, old_closure = dag.closure(), old_dag.closure()
    kept_orderings = [e for e in old_closure if e[0] not in gold and e[1] not in gold]
    return dict(
        method=name, nodes=len(dag.g), edges=len(dag.g.edges),
        affected=prf(predicted, gold), gold_affected=len(gold), predicted_affected=len(predicted),
        unaffected_edges_retained=retained / len(kept_unaffected) if kept_unaffected else 1.0,
        unaffected_orderings_retained=(sum(e in closure for e in kept_orderings) / len(kept_orderings)
                                       if kept_orderings else 1.0),
        stale_nodes=sum(1 for n in dag.g if not _current(dag, n, version)),
        retired_on_record=len(dag.retired), renames_absorbed=len(report.get("absorbed", ())),
        validation={k: len(v) for k, v in dag.validate().items()},
        learner={**learner_report(old_dag, dag, reference.g, version),
                 **judge_confirmed_violations(old_dag, dag, reference)},
        judge_edge_precision=judge_precision(dag),
        anomalies=report.get("anomalies"), **ledger.to_dict(),
    )


def graph_changes(old_dag, new_dag):
    """Nodes a rebuild changed, found by comparing graphs because a rebuild has no diff to apply."""
    return sorted((set(old_dag.g) ^ set(new_dag.g))
                  | {n for e in set(old_dag.g.edges) ^ set(new_dag.g.edges) for n in e})


def compare_methods(label, old_q, new_q, old_version, version):
    """Revision agent against full rebuild and append-only. Logs and returns one row per method."""
    OUT.mkdir(parents=True, exist_ok=True)
    diff = diff_questions(old_q, new_q)
    print("diff", {k: len(v) for k, v in diff.items()}, flush=True)
    (OUT / f"{label}_diff.json").write_text(json.dumps(diff, indent=1))

    base_ledger = Ledger()
    old_dag, old_report = build_from_questions(old_q, old_version, ledger=base_ledger)
    old_dag.save(OUT / f"{label}_{old_version}.json")
    print(old_version, len(old_dag.g), len(old_dag.g.edges), old_report, base_ledger.to_dict(), flush=True)

    rebuild_ledger = Ledger()
    rebuilt, _ = build_from_questions(new_q, version, ledger=rebuild_ledger)
    rebuilt.save(OUT / f"{label}_{version}_rebuild.json")
    gold = gold_affected(old_dag, diff, rebuilt)
    rows = [evaluate("full_rebuild", rebuilt, dict(affected=graph_changes(old_dag, rebuilt), anomalies=None),
                     rebuild_ledger, old_dag, diff, rebuilt, version)]

    # Noise floor: the same rebuild with only the candidate list order reversed.
    ledger = Ledger()
    perturbed, _ = build_from_questions(new_q, version, ledger=ledger, reverse_candidates=True)
    perturbed.save(OUT / f"{label}_{version}_rebuild_reversed.json")
    rows.append(evaluate("rebuild_reversed_list", perturbed, dict(affected=graph_changes(old_dag, perturbed), anomalies=None),
                         ledger, old_dag, diff, rebuilt, version))

    ledger = Ledger()
    revised, report = revise(old_dag, diff, new_q, version, ledger)
    revised.save(OUT / f"{label}_{version}_revised.json")
    rows.append(evaluate("revision_agent", revised, report, ledger, old_dag, diff, rebuilt, version))

    ledger = Ledger()
    appended, report = append_only(old_dag, diff, new_q, version, ledger)
    appended.save(OUT / f"{label}_{version}_append.json")
    rows.append(evaluate("append_only", appended, report, ledger, old_dag, diff, rebuilt, version))

    for row in rows:
        log_run("revision", dict(stage=label, method=row["method"], prompt_version=DEFAULT_EDGE_PROMPT, model=MUSE_MODEL), row)
        print(json.dumps(row), flush=True)
    print("gold affected", len(gold))
    return rows


def main():
    old_q, new_q = load_civics()
    assert (len(old_q), len(new_q)) == (100, 128), (len(old_q), len(new_q))
    compare_methods("civics_uncapped", old_q, new_q, "2008", "2025")


if __name__ == "__main__":
    main()
