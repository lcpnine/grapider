"""Normalized schema and dataset loaders.

Direction convention, everywhere: an edge (prereq_id, dependent_id, label) with
label 1 means prereq_id must be learned before dependent_id.
"""
import csv
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"
UNLOCK_ENV = "PREREQ_RUN_TEST"


def guard_test_path(path):
    """Refuse to read data/test unless the test run was explicitly unlocked."""
    p = Path(path).resolve()
    if p.is_relative_to(TEST_DIR) and os.environ.get(UNLOCK_ENV) != "1":
        raise PermissionError(f"{p} is test data; locked until 'run test'")


def canon(name):
    """Canonical concept id, so one concept keeps one id across datasets."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    if len(words) > 1 and words[-1] == "s":
        words.pop()
    if words:
        last = words[-1]
        if last.endswith("ies") and len(last) > 4:
            words[-1] = last[:-3] + "y"
        elif last.endswith("s") and not last.endswith(("ss", "is", "us")) and len(last) > 3:
            words[-1] = last[:-1]
    return "_".join(words)


def normalize(names, rows, source):
    """Build (concepts, edges, stats) from {raw_id: name} and (prereq, dependent, label) rows.

    Dropped and counted: unknown ids, self-loops, pairs with conflicting labels,
    and pairs labeled 1 in both directions (they cannot be in a DAG).
    """
    stats = Counter(rows_in=0)
    concepts = {}
    raw_to_id = {}
    for raw, name in names.items():
        cid = canon(name)
        raw_to_id[raw] = cid
        concepts.setdefault(cid, dict(id=cid, name=name, description="", source=source))
    labels = {}
    conflicted = set()
    for prereq, dependent, label in rows:
        stats["rows_in"] += 1
        if prereq not in raw_to_id or dependent not in raw_to_id:
            stats["unknown_id"] += 1
            continue
        pair = (raw_to_id[prereq], raw_to_id[dependent])
        if pair[0] == pair[1]:
            stats["self_loop"] += 1
            continue
        label = int(label)
        if label not in (0, 1):
            raise ValueError(f"bad label {label!r} in {source}")
        if pair in labels and labels[pair] != label:
            conflicted.add(pair)
        elif pair in labels:
            stats["duplicate"] += 1
        labels[pair] = label
    stats["conflict"] = len(conflicted)
    both = {p for p, v in labels.items() if v == 1 and labels.get(p[::-1]) == 1}
    stats["both_directions"] = len(both) // 2
    edges = [
        dict(prereq_id=p, dependent_id=d, label=v, source=source)
        for (p, d), v in sorted(labels.items())
        if (p, d) not in conflicted and (p, d) not in both
    ]
    stats["edges_out"] = len(edges)
    stats["positive_out"] = sum(e["label"] for e in edges)
    return list(concepts.values()), edges, dict(stats)


def _read_csv(path):
    with open(path, encoding="utf8", errors="replace", newline="") as f:
        return [r for r in csv.reader(f) if r]


def load_lb1(root):
    """LB1 rows are (prereq, dependent, label)."""
    names = {r[0]: r[1] for r in _read_csv(Path(root) / "208topics.csv")}
    return normalize(names, _read_csv(Path(root) / "prerequisite_annotation.csv"), "lb1")


def load_lb2(root):
    """LB2 rows are (prereq, dependent, label); topics are id|name."""
    data = Path(root) / "LectureBank2" / "data"
    names = {}
    for line in open(data / "322topics_final.tsv", encoding="utf8"):
        if "|" in line:
            raw, name = line.rstrip("\n").split("|", 1)
            names[raw] = name
    return normalize(names, _read_csv(data / "final_new_annotation.csv"), "lb2")


def load_ucd(path):
    """UCD 'A,B' means B is a prerequisite of A, so the columns are swapped here."""
    guard_test_path(path)
    rows = [(b, a, 1) for a, b in (r[:2] for r in _read_csv(path))]
    names = {x: x for r in rows for x in r[:2]}
    return normalize(names, rows, "ucd")


def load_dbe(path):
    """DBE-KT22 graphml edges point dependent->prereq, so they are reversed here."""
    guard_test_path(path)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    graph = ET.parse(path).getroot()
    names = {n.get("id"): n.get("id") for n in graph.iterfind(".//g:node", ns)}
    rows = [(e.get("target"), e.get("source"), 1) for e in graph.iterfind(".//g:edge", ns)]
    return normalize(names, rows, "dbe")
