from prereq.civics import diff_questions, parse_questions
from prereq.dag import PrereqDAG
from prereq.revision import gold_affected, plan_violations, retire_node, revise


def q(version, number, question, answer):
    return dict(id=f"{version}:q{number}", number=number, question=question, answers=[answer],
                text=f"{question} Answers: {answer}")


def dag_of(*edges):
    dag = PrereqDAG()
    for u, v in edges:
        dag.add_node(u, provenance=[f"old:{u}"], valid_versions=["old"])
        dag.add_node(v, provenance=[f"old:{v}"], valid_versions=["old"])
        dag.add_edge(u, v, "needed")
    return dag


def test_parse_questions_2008_style():
    text = "AMERICAN GOVERNMENT\n\n1. What is the supreme law of the land?\n\n. the Constitution\n\n2. What does it do?\n\n. sets up the government\n. protects rights\n"
    parsed = parse_questions(text, "2008")
    assert [p["id"] for p in parsed] == ["2008:q1", "2008:q2"]
    assert parsed[1]["answers"] == ["sets up the government", "protects rights"]


def test_parse_questions_ignores_out_of_sequence_numbers():
    parsed = parse_questions("1. First question?\n   • yes\n7. Not a question\n2. Second question?\n   • no\n", "v")
    assert [p["number"] for p in parsed] == [1, 2]


def test_diff_classifies_all_four_cases():
    old = [q("o", 1, "What is the capital of the country?", "x"), q("o", 2, "Who wrote the national anthem?", "y"),
           q("o", 3, "Name one river in the region today?", "z")]
    new = [q("n", 1, "What is the capital of the country?", "x"), q("n", 2, "Who wrote the national anthem?", "CHANGED"),
           q("n", 3, "How many senators sit in congress now?", "100")]
    diff = diff_questions(old, new)
    assert diff["unchanged"] == [("o:q1", "n:q1")]
    assert diff["modified"] == [("o:q2", "n:q2")]
    assert diff["removed"] == ["o:q3"] and diff["added"] == ["n:q3"]


def test_retire_node_bridges_parents_to_children():
    dag = dag_of(("a", "b"), ("b", "c"))
    anomalies = dict(broken_path_bridged=0)
    retire_node(dag, "b", anomalies)
    assert dag.g.has_edge("a", "c") and anomalies["broken_path_bridged"] == 1
    assert "\n" not in dag.g["a"]["c"]["reason"]


def test_revise_with_no_changes_keeps_graph_and_makes_no_calls():
    from prereq.llm import Ledger
    dag = PrereqDAG()
    dag.add_node("a", provenance=["o:q1"], valid_versions=["o"])
    dag.add_node("b", provenance=["o:q2"], valid_versions=["o"])
    dag.add_edge("a", "b", "needed")
    diff = dict(unchanged=[("o:q1", "n:q1"), ("o:q2", "n:q2")], modified=[], removed=[], added=[])
    ledger = Ledger()
    revised, report = revise(dag, diff, [], "n", ledger)
    assert ledger.calls == 0 and report["affected"] == []
    assert revised.g.has_edge("a", "b")
    assert revised.g.nodes["a"]["provenance"] == {"n:q1"}
    assert revised.g.nodes["a"]["valid_versions"] == {"o", "n"}


def test_gold_affected_uses_both_versions():
    old = PrereqDAG()
    old.add_node("gone", provenance=["o:q1"])
    old.add_node("same", provenance=["o:q2"])
    new = PrereqDAG()
    new.add_node("fresh", provenance=["n:q3"])
    new.add_node("same", provenance=["n:q2"])
    diff = dict(unchanged=[("o:q2", "n:q2")], modified=[], removed=["o:q1"], added=["n:q3"])
    assert gold_affected(old, diff, new) == {"gone", "fresh"}


def test_plan_violations_counts_wrong_order_and_missing_prereqs():
    ref = dag_of(("a", "b"), ("b", "c")).g
    assert plan_violations(["a", "b", "c"], set(), ref) == 0
    assert plan_violations(["b", "a", "c"], set(), ref) == 1
    assert plan_violations(["c"], {"a"}, ref) == 1
    assert plan_violations(["c"], {"a", "b"}, ref) == 0


def test_parse_strips_page_footer():
    parsed = parse_questions("1. What is Veterans Day?\n   • A holiday 19 of 19        uscis.gov/citizenship\n", "v")
    assert parsed[0]["answers"] == ["A holiday"]


def test_append_only_renames_unchanged_provenance_and_keeps_removed():
    from prereq.llm import Ledger
    from prereq.revision import append_only
    dag = PrereqDAG()
    dag.add_node("kept", provenance=["o:q1"])
    dag.add_node("old", provenance=["o:q2"])
    diff = dict(unchanged=[("o:q1", "n:q1")], modified=[], removed=["o:q2"], added=[])
    result, _ = append_only(dag, diff, [], "n", Ledger())
    assert result.g.nodes["kept"]["provenance"] == {"n:q1"}
    assert result.g.nodes["old"]["provenance"] == {"o:q2"}
