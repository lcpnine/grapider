import json

import pytest

from prereq import test_report, test_run
from prereq.dag import PrereqDAG
from prereq.schema import ROOT, load_lbcd


def test_refuses_without_the_unlock(monkeypatch):
    monkeypatch.delenv("PREREQ_RUN_TEST", raising=False)
    with pytest.raises(PermissionError):
        test_run.main()


def test_lbcd_loader_is_locked_before_anything_else(monkeypatch):
    monkeypatch.delenv("PREREQ_RUN_TEST", raising=False)
    with pytest.raises(PermissionError):
        load_lbcd(ROOT / "data" / "test" / "raw" / "lecturebankcd", "BIO")


def test_closed_world_universe_is_every_ordered_pair():
    concepts = [dict(id="a"), dict(id="b")]
    assert test_run.closed_world_universe(concepts) == {("a", "b"), ("b", "a")}


def test_finished_reads_logged_datasets_so_results_cannot_be_rerolled(monkeypatch, tmp_path):
    log = tmp_path / "test.jsonl"
    monkeypatch.setattr(test_run, "LOG", log)
    assert test_run.finished() == set()
    log.write_text(json.dumps(dict(config=dict(dataset="ucd"))) + "\n")
    assert test_run.finished() == {"ucd"}


def test_report_scores_raw_labels_and_exposes_closure_contradictions():
    concepts = [dict(id=x) for x in "abc"]
    edges = [dict(prereq_id="a", dependent_id="b", label=1), dict(prereq_id="b", dependent_id="c", label=1),
             dict(prereq_id="a", dependent_id="c", label=0)]
    dag = PrereqDAG()
    for x in "abc":
        dag.add_node(x)
    dag.add_edge("a", "b", "needed")
    result = test_report.report(concepts, edges, dag, closed_world=False)
    assert result["negatives_closure_calls_positive"] == 1
    assert result["direct"]["precision"] == 1.0 and result["direct"]["recall"] == 0.5
    assert result["direct_flipped"]["scored"] == 0
