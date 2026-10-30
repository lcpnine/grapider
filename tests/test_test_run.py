import json

import pytest

from prereq import test_run
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
