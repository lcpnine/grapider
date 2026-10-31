"""Check the finished test graphs against raw labels. Added after the test run, so it tunes nothing.

Reachability stays the primary score. This shows what it hides when gold labels form cycles,
and whether a dataset can tell the predicted direction from its reverse. No model calls.

Usage: PREREQ_RUN_TEST=1 python -m prereq.test_report
"""
import csv
import json

import networkx as nx

from .dag import PrereqDAG
from .llm import log_run
from .pipeline import gold_reachability
from .revision import OUT, prf
from .schema import canon, load_dbe, load_lbcd, load_ucd
from .test_run import RAW, closed_world_universe


def scored(predicted, positives, negatives):
    pool = positives | negatives
    in_scope = predicted & pool
    return {**prf(in_scope, positives), "scored": len(in_scope), "predicted": len(predicted),
            "lift": prf(in_scope, positives)["precision"] / (len(positives) / len(pool))}


def report(concepts, edges, dag, closed_world):
    labels = {(e["prereq_id"], e["dependent_id"]): e["label"] for e in edges}
    positives = {p for p, label in labels.items() if label == 1}
    negatives = {p for p, label in labels.items() if label == 0}
    if closed_world:
        negatives |= closed_world_universe(concepts) - positives
    gold = nx.DiGraph(positives)
    cyclic = [c for c in nx.strongly_connected_components(gold) if len(c) > 1]
    reach, _ = gold_reachability(edges)
    closure, direct = dag.closure(), set(dag.g.edges)
    return dict(
        labeled_positive=len(positives), labeled_negative=len(negatives),
        labeled_prevalence=len(positives) / (len(positives) + len(negatives)),
        gold_cyclic_groups=len(cyclic), gold_largest_cycle=max(map(len, cyclic), default=0),
        gold_closure=len(reach), negatives_closure_calls_positive=len(reach & negatives),
        closure=scored(closure, positives, negatives), direct=scored(direct, positives, negatives),
        closure_flipped=scored({(b, a) for a, b in closure}, positives, negatives),
        direct_flipped=scored({(b, a) for a, b in direct}, positives, negatives),
    )


def main():
    datasets = {
        "lbcd_bio": (load_lbcd(RAW / "lecturebankcd", "BIO"), False),
        "lbcd_cv": (load_lbcd(RAW / "lecturebankcd", "CV"), False),
        "ucd": (load_ucd(RAW / "ucd" / "cs_annotations.tsv"), False),
        "dbe": (load_dbe(RAW / "dbe" / "data" / "wrapup" / "DBE-KT22.graphml"), True),
    }
    for name, ((concepts, edges, _), closed_world) in datasets.items():
        dag = PrereqDAG.load(OUT / f"test_{name}.json")
        result = report(concepts, edges, dag, closed_world)
        if name == "ucd":
            with open(RAW / "ucd" / "cs_preqs.csv", encoding="utf8", errors="replace", newline="") as f:
                preqs = {(canon(b.replace("_", " ")), canon(a.replace("_", " "))) for a, b, *_ in csv.reader(f)}
            labeled = {(e["prereq_id"], e["dependent_id"]) for e in edges}
            result["closure_vs_cs_preqs"] = scored(dag.closure(), preqs, labeled - preqs)
        log_run("test_report", dict(dataset=name, stage="raw_labels"), result)
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
