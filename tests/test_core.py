import pytest

from prereq.dag import PrereqDAG
from prereq.schema import ROOT, canon, guard_test_path, load_dbe, load_ucd, normalize
from prereq.splits import make_splits


def chain(*edges):
    dag = PrereqDAG()
    for u, v in edges:
        dag.add_node(u)
        dag.add_node(v)
        dag.add_edge(u, v, "needed")
    return dag


def test_canon_merges_variants():
    assert canon("Probabilities") == canon("probability")
    assert canon("CNN's") == canon("cnn")
    assert canon("Bayes Theorem") == "bayes_theorem"
    assert canon("analysis") == "analysis"


def test_normalize_drops_and_counts_bad_rows():
    names = {"1": "a", "2": "b", "3": "c", "4": "d"}
    rows = [("1", "2", 1), ("1", "2", 1), ("1", "1", 0), ("9", "1", 0),
            ("2", "3", 0), ("2", "3", 1), ("3", "4", 1), ("4", "3", 1), ("1", "3", 0)]
    _, edges, stats = normalize(names, rows, "x")
    assert {(e["prereq_id"], e["dependent_id"]): e["label"] for e in edges} == {("a", "b"): 1, ("a", "c"): 0}
    assert (stats["duplicate"], stats["self_loop"], stats["unknown_id"]) == (1, 1, 1)
    assert (stats["conflict"], stats["both_directions"]) == (1, 1)


def test_ucd_second_column_is_the_prereq(tmp_path):
    f = tmp_path / "preqs.csv"
    f.write_text("advanced,intro\n")
    _, edges, _ = load_ucd(f)
    assert (edges[0]["prereq_id"], edges[0]["dependent_id"], edges[0]["label"]) == ("intro", "advanced", 1)


def test_dbe_edges_are_reversed(tmp_path):
    f = tmp_path / "g.graphml"
    f.write_text('<graphml xmlns="http://graphml.graphdrawing.org/xmlns"><graph edgedefault="directed">'
                 '<node id="advanced"/><node id="intro"/><edge source="advanced" target="intro"/>'
                 "</graph></graphml>")
    _, edges, _ = load_dbe(f)
    assert (edges[0]["prereq_id"], edges[0]["dependent_id"]) == ("intro", "advanced")


def test_test_data_is_locked(monkeypatch):
    monkeypatch.delenv("PREREQ_RUN_TEST", raising=False)
    with pytest.raises(PermissionError):
        guard_test_path(ROOT / "data" / "test" / "raw" / "anything")
    with pytest.raises(PermissionError):
        load_ucd(ROOT / "data" / "test" / "raw" / "ucd" / "cs_preqs.csv")


def _dataset(names, source):
    rows = [(a, b, 1 if a < b else 0) for a in names for b in names if a != b]
    return normalize({n: n for n in names}, rows, source)


def test_splits_share_no_concepts_even_through_lb2():
    lb1 = _dataset([f"c{i}" for i in range(50)], "lb1")
    lb2 = _dataset([f"c{i}" for i in range(25, 75)], "lb2")
    judge, system, report = make_splits(lb1, lb2)
    judge_ids = {c["id"] for c in judge["concepts"]}
    system_ids = {c["id"] for c in system["concepts"]}
    assert len(judge_ids) == 10 and not judge_ids & system_ids
    for e in system["edges"]:
        assert not {e["prereq_id"], e["dependent_id"]} & judge_ids
    for e in judge["edges"]:
        assert {e["prereq_id"], e["dependent_id"]} <= judge_ids
    assert report["crossing_pairs_dropped"] > 0
    assert make_splits(lb1, lb2)[0] == judge


def test_edge_needs_one_line_reason_and_known_nodes():
    dag = chain(("a", "b"))
    for bad in ("", "  ", "two\nlines"):
        with pytest.raises(ValueError):
            dag.add_edge("a", "b", bad)
    with pytest.raises(KeyError):
        dag.add_edge("a", "missing", "why")
    with pytest.raises(ValueError):
        dag.add_edge("a", "a", "why")


def test_provenance_and_versions_merge():
    dag = PrereqDAG()
    dag.add_node("a", provenance=["q1"], valid_versions=["2008"])
    dag.add_node("a", provenance=["q2"], valid_versions=["2025"])
    assert dag.g.nodes["a"]["provenance"] == {"q1", "q2"}
    assert dag.g.nodes["a"]["valid_versions"] == {"2008", "2025"}


def test_clean_graph_passes_all_validators():
    assert not any(chain(("a", "b"), ("b", "c")).validate().values())


def test_validate_finds_cycle():
    assert chain(("a", "b"), ("b", "a")).validate()["cycles"]


def test_validate_finds_too_many_parents():
    dag = chain(("a", "x"), ("b", "x"), ("c", "x"))
    assert dag.validate()["too_many_parents"] == []
    dag.add_node("d")
    dag.add_edge("d", "x", "needed")
    assert dag.validate()["too_many_parents"] == ["x"]


def test_validate_finds_orphan():
    dag = chain(("a", "b"))
    dag.add_node("lonely")
    assert dag.validate()["orphans"] == ["lonely"]


def test_transitive_reduction_keeps_reachability():
    dag = chain(("a", "b"), ("b", "c"), ("a", "c"))
    assert dag.validate()["redundant_edges"] == [("a", "c")]
    before = dag.closure()
    assert dag.transitive_reduce() == [("a", "c")]
    assert dag.closure() == before
    assert not dag.g.has_edge("a", "c")


def test_reduce_refuses_cycles():
    with pytest.raises(ValueError):
        chain(("a", "b"), ("b", "a")).transitive_reduce()


def test_path_is_ordered_and_skips_known():
    dag = chain(("a", "b"), ("b", "d"), ("c", "d"), ("d", "e"))
    dag.add_node("unrelated")
    path = dag.path_to("d")
    assert set(path) == {"a", "b", "c", "d"}
    assert path.index("a") < path.index("b") < path.index("d")
    assert dag.path_to("d", known=["a", "b"]) == ["c", "d"]


def test_json_round_trip(tmp_path):
    dag = chain(("a", "b"))
    dag.add_node("a", provenance=["q1"], valid_versions=["v1"])
    dag.save(tmp_path / "d.json")
    assert PrereqDAG.load(tmp_path / "d.json").to_json() == dag.to_json()
