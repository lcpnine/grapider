"""Build pipeline (Muse). Fixed-concept mode, and extract->match->edges->validate->path for documents.

Usage: python -m prereq.pipeline grid
"""
import difflib
import json
import sys

import networkx as nx

from .dag import MAX_PARENTS, PrereqDAG
from .llm import MUSE_MODEL, Ledger, ask_json, log_run, pmap
from .schema import canon
from .splits import SPLIT_DIR

EDGE_OUTPUT = ('Reply with JSON only: {"prereqs": [{"name": "<exact name from the list>", '
               '"reason": "<one line>"}]}. Use an empty list if there are none.')
EDGE_PROMPTS = {
    "e1_direct": f"You build a prerequisite graph. Given a target concept and a list of concepts, pick up to "
                 f"{MAX_PARENTS} from the list that are its direct prerequisites. " + EDGE_OUTPUT,
    "e2_strict": f"You build a prerequisite graph. Given a target concept and a list of concepts, pick up to "
                 f"{MAX_PARENTS} from the list that a learner must understand before the target. Being related "
                 f"is not enough, and do not pick concepts that build on the target. Many foundational concepts "
                 f"have no prerequisites in the list. " + EDGE_OUTPUT,
    "e3_chain": f"You build a prerequisite graph where long chains matter. Given a target concept and a list of "
                f"concepts, pick up to {MAX_PARENTS} from the list that are the closest immediate steps before "
                f"the target: choose the most advanced concepts the target directly builds on, not distant "
                f"fundamentals they already rely on. Foundational concepts may have none. " + EDGE_OUTPUT,
}
EXTRACT_SYSTEM = ('Extract the 1 to 3 core study concepts a learner must know to answer this question. Use short, '
                  'general noun phrases, not the question text and not answers that change over time. '
                  'Reply with JSON only: {"concepts": ["..."]}.')
MERGE_SYSTEM = ('Two concept labels come from the same study material. Decide if they mean the same concept. '
                'Reply with JSON only: {"same": true or false}.')
AUTO_MERGE, ASK_MERGE = 0.92, 0.80
DEFAULT_EDGE_PROMPT = "e2_strict"


def pick_prereqs(target, names, prompt_id, ledger=None):
    """Return [(prereq_name, reason)] chosen by Muse, limited to the given names."""
    others = sorted(n for n in names if n != target)
    user = f'Target: "{target}"\nConcepts:\n' + "\n".join(f"- {n}" for n in others)
    out = ask_json(EDGE_PROMPTS[prompt_id], user, ledger)
    known = {canon(n): n for n in others}
    picks = []
    for p in out.get("prereqs", []):
        name = known.get(canon(str(p.get("name", ""))))
        if name and name not in [x[0] for x in picks]:
            picks.append((name, " ".join(str(p.get("reason") or "selected by model").split())))
    return picks[:MAX_PARENTS]


def break_cycles(dag, judge_cycle=None):
    """Remove one edge per cycle until acyclic and return the removed edges.

    judge_cycle(cycle_edges) may choose the edge. By default the edge whose prereq is
    least foundational (lowest out-degree) goes, with ties broken by name.
    """
    removed = []
    while True:
        try:
            cycle = [(u, v) for u, v, *_ in nx.find_cycle(dag.g)]
        except nx.NetworkXNoCycle:
            return removed
        edge = judge_cycle(cycle) if judge_cycle else None
        if edge not in cycle:
            edge = min(cycle, key=lambda e: (dag.g.out_degree(e[0]), e))
        dag.g.remove_edge(*edge)
        removed.append(edge)


def build_edges(dag, prompt_id=DEFAULT_EDGE_PROMPT, ledger=None, targets=None, judge_cycle=None):
    """Ask for parents of each target, then enforce the DAG rules. Returns a small report."""
    names = {a["name"]: n for n, a in dag.g.nodes(data=True)}
    targets = list(names) if targets is None else targets
    picked = pmap(lambda t: pick_prereqs(t, list(names), prompt_id, ledger), targets)
    for target, picks in zip(targets, picked):
        for name, reason in picks:
            dag.add_edge(names[name], names[target], reason)
    cycles = break_cycles(dag, judge_cycle)
    reduced = dag.transitive_reduce()
    return dict(cycle_edges_removed=len(cycles), redundant_edges_removed=len(reduced))


def extract_concepts(question, ledger=None):
    out = ask_json(EXTRACT_SYSTEM, question, ledger)
    return [str(c).strip() for c in out.get("concepts", []) if str(c).strip()]


def match_concept(name, existing, ledger=None):
    """Return the id this name merges into, or None. Only close-but-unsure pairs reach the LLM."""
    cid = canon(name)
    if cid in existing:
        return cid
    if not existing:
        return None
    best = max(sorted(existing), key=lambda e: difflib.SequenceMatcher(None, cid, e).ratio())
    score = difflib.SequenceMatcher(None, cid, best).ratio()
    if score >= AUTO_MERGE:
        return best
    if score >= ASK_MERGE:
        out = ask_json(MERGE_SYSTEM, f'A: "{name}"\nB: "{existing[best]}"', ledger)
        return best if out.get("same") else None
    return None


def add_questions(dag, questions, version, ledger=None):
    """Extract concepts and attach provenance. Returns the ids of nodes these questions touch."""
    extracted = pmap(lambda q: extract_concepts(q["text"], ledger), questions)
    names = {n: a["name"] for n, a in dag.g.nodes(data=True)}
    touched = set()
    for q, concepts in zip(questions, extracted):
        for concept in concepts:
            cid = match_concept(concept, names, ledger) or canon(concept)
            if not cid:
                continue
            names.setdefault(cid, concept)
            dag.add_node(cid, concept, [q["id"]], [version])
            touched.add(cid)
    return touched


def build_from_questions(questions, version, prompt_id=DEFAULT_EDGE_PROMPT, ledger=None):
    """Full document pipeline: extract -> match -> edges -> validate."""
    dag = PrereqDAG()
    add_questions(dag, questions, version, ledger)
    report = build_edges(dag, prompt_id, ledger)
    report["validation"] = {k: len(v) for k, v in dag.validate().items()}
    return dag, report


def gold_reachability(edges):
    """Gold reachable pairs, and the scored universe: labeled pairs plus gold reachable pairs.

    A pair labeled 1 by either dataset counts as positive.
    """
    g = nx.DiGraph((e["prereq_id"], e["dependent_id"]) for e in edges if e["label"] == 1)
    reach = {(u, v) for u, v in nx.transitive_closure(g, reflexive=False).edges if u != v}
    return reach, reach | {(e["prereq_id"], e["dependent_id"]) for e in edges}


def score_reachability(dag, reach, universe):
    predicted = dag.closure()
    in_scope = predicted & universe
    tp = len(in_scope & reach)
    precision = tp / len(in_scope) if in_scope else 0.0
    recall = tp / len(reach) if reach else 0.0
    return dict(
        precision=precision, recall=recall,
        f1=2 * precision * recall / (precision + recall) if tp else 0.0,
        predicted_scored=len(in_scope), predicted_unlabeled=len(predicted - universe),
        gold_reachable=len(reach), universe=len(universe), prevalence=len(reach) / len(universe),
    )


def fixed_concept_dag(split, prompt_id, ledger=None):
    dag = PrereqDAG()
    for c in split["concepts"]:
        dag.add_node(c["id"], c["name"], [c["source"]])
    report = build_edges(dag, prompt_id, ledger)
    report["validation"] = {k: len(v) for k, v in dag.validate().items()}
    return dag, report


def grid():
    split = json.loads((SPLIT_DIR / "system_dev.json").read_text())
    reach, universe = gold_reachability(split["edges"])
    results = {}
    for prompt_id in EDGE_PROMPTS:
        ledger = Ledger()
        dag, report = fixed_concept_dag(split, prompt_id, ledger)
        result = {**score_reachability(dag, reach, universe), **report, **ledger.to_dict(), "edges": len(dag.g.edges)}
        log_run("build", dict(stage="fixed_grid", split="system_dev", prompt_version=prompt_id, model=MUSE_MODEL), result)
        results[prompt_id] = result
        print(prompt_id, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in result.items()}), flush=True)
    return results


if __name__ == "__main__":
    {"grid": grid}[sys.argv[1]]()
