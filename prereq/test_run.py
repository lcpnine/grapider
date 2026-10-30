"""One-shot test run with the frozen dev configuration. Nothing here tunes anything.

Usage: PREREQ_RUN_TEST=1 python -m prereq.test_run

A dataset already logged to runs/test.jsonl is never scored again, so a crash can resume
but a finished result cannot be re-rolled. UCD and DBE-KT22 list positives only, so every
unlisted ordered pair counts as a negative, which makes their precision a lower bound.
"""
import json
import os
import subprocess

from .civics import HAM, load_ham
from .llm import MUSE_MODEL, RUNS, Ledger, log_run
from .pipeline import DEFAULT_EDGE_PROMPT, fixed_concept_dag, gold_reachability, score_reachability
from .revision import compare_methods, judge_precision
from .schema import TEST_DIR, UNLOCK_ENV, load_dbe, load_lbcd, load_ucd

RAW = TEST_DIR / "raw"
LOG = RUNS / "test.jsonl"
MIN_HAM_QUESTIONS = 300


def closed_world_universe(concepts):
    ids = [c["id"] for c in concepts]
    return {(a, b) for a in ids for b in ids if a != b}


def git_state():
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout)
    return dict(commit=head, dirty=dirty)


def finished():
    if not LOG.exists():
        return set()
    return {json.loads(line)["config"]["dataset"] for line in LOG.open()}


def score_fixed(name, concepts, edges, closed_world, git):
    split = dict(concepts=concepts)
    reach, universe = gold_reachability(edges)
    if closed_world:
        universe |= closed_world_universe(concepts)
    ledger = Ledger()
    dag, report = fixed_concept_dag(split, DEFAULT_EDGE_PROMPT, ledger)
    result = {**score_reachability(dag, reach, universe), **report, **ledger.to_dict(), "edges": len(dag.g.edges),
              "concepts": len(concepts), "judge_edge_precision": judge_precision(dag)}
    log_run("test", dict(dataset=name, stage="fixed_concept", closed_world=closed_world, git=git,
                         prompt_version=DEFAULT_EDGE_PROMPT, model=MUSE_MODEL), result)
    print(name, json.dumps(result), flush=True)


def run_ham(git):
    old_q, new_q = load_ham()
    assert min(len(old_q), len(new_q)) >= MIN_HAM_QUESTIONS, (len(old_q), len(new_q))
    rows = compare_methods("ham", old_q, new_q, "2022", "2026")
    log_run("test", dict(dataset="ham", stage="revision", git=git, prompt_version=DEFAULT_EDGE_PROMPT,
                         model=MUSE_MODEL), dict(rows=rows))


def main():
    if os.environ.get(UNLOCK_ENV) != "1":
        raise PermissionError("locked until 'run test'")
    git = git_state()
    if git["dirty"]:
        raise RuntimeError("commit first, so the result points at the exact code that produced it")
    missing = [p for p in (HAM / "tech_2022_2026.docx", HAM / "tech_2026_2030.docx") if not p.exists()]
    if missing:
        raise FileNotFoundError(f"download these first: {[str(p) for p in missing]}")
    fixed = {
        "lbcd_bio": lambda: (load_lbcd(RAW / "lecturebankcd", "BIO"), False),
        "lbcd_cv": lambda: (load_lbcd(RAW / "lecturebankcd", "CV"), False),
        "ucd": lambda: (load_ucd(RAW / "ucd" / "cs_preqs.csv"), True),
        "dbe": lambda: (load_dbe(RAW / "dbe" / "data" / "wrapup" / "DBE-KT22.graphml"), True),
    }
    done = finished()
    for name, load in fixed.items():
        if name not in done:
            (concepts, edges, stats), closed_world = load()
            print(name, "normalized", stats, flush=True)
            score_fixed(name, concepts, edges, closed_world, git)
    if "ham" not in done:
        run_ham(git)


if __name__ == "__main__":
    main()
