from prereq import pipeline
from prereq.dag import PrereqDAG
from prereq.pipeline import break_cycles, gold_reachability, match_concept, score_reachability


def edge(a, b, label=1):
    return dict(prereq_id=a, dependent_id=b, label=label)


def test_gold_reachability_includes_transitive_pairs():
    reach, universe = gold_reachability([edge("a", "b"), edge("b", "c"), edge("c", "a", 0)])
    assert reach == {("a", "b"), ("b", "c"), ("a", "c")}
    assert ("c", "a") in universe and ("c", "a") not in reach


def test_score_counts_indirect_path_as_a_hit():
    dag = PrereqDAG()
    for n in "abc":
        dag.add_node(n)
    dag.add_edge("a", "b", "x")
    dag.add_edge("b", "c", "x")
    reach, universe = gold_reachability([edge("a", "c")])
    result = score_reachability(dag, reach, universe)
    assert result["recall"] == 1.0 and result["precision"] == 1.0
    assert result["predicted_unlabeled"] == 2


def test_break_cycles_prefers_removing_least_foundational_prereq():
    dag = PrereqDAG()
    for n in "abcd":
        dag.add_node(n)
    dag.add_edge("a", "b", "x")
    dag.add_edge("a", "d", "x")
    dag.add_edge("b", "a", "x")
    assert break_cycles(dag) == [("b", "a")]


def test_match_exact_and_no_llm_for_distant_names(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_json", lambda *a, **k: 1 / 0)
    existing = {"bill_of_rights": "Bill of Rights"}
    assert match_concept("Bill of Rights", existing) == "bill_of_rights"
    assert match_concept("photosynthesis", existing) is None


def test_match_asks_llm_only_when_ambiguous(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "ask_json", lambda *a, **k: calls.append(a) or {"same": False})
    assert match_concept("federal courts", {"federal_courtesy": "federal courtesy"}) is None
    assert len(calls) == 1
