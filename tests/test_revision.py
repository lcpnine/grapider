import zipfile

import pytest

from prereq import pipeline, revision
from prereq.civics import diff_questions, docx_text, load_ham, parse_ham, parse_questions
from prereq.dag import PrereqDAG
from prereq.llm import Ledger
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
    retire_node(dag, "b", anomalies, "new")
    assert dag.g.has_edge("a", "c") and anomalies["broken_path_bridged"] == 1
    assert "b" not in dag.g
    assert dag.retired["b"]["retired_in"] == "new" and dag.retired["b"]["parents"] == ["a"]
    assert "\n" not in dag.g["a"]["c"]["reason"]


def test_revise_with_no_changes_keeps_graph_and_makes_no_calls():
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
    from prereq.revision import append_only
    dag = PrereqDAG()
    dag.add_node("kept", provenance=["o:q1"])
    dag.add_node("old", provenance=["o:q2"])
    diff = dict(unchanged=[("o:q1", "n:q1")], modified=[], removed=["o:q2"], added=[])
    result, _ = append_only(dag, diff, [], "n", Ledger())
    assert result.g.nodes["kept"]["provenance"] == {"n:q1"}
    assert result.g.nodes["old"]["provenance"] == {"o:q2"}


class FakeMuse:
    """Answers each prompt type from fixed tables, and records what the agent asked."""

    def __init__(self, concepts=None, parents=None, same=(), keep="current"):
        self.concepts, self.parents, self.same, self.keep = concepts or {}, parents or {}, set(same), keep
        self.asked = []

    def __call__(self, system, user, ledger=None, model=None):
        self.asked.append(system)
        if ledger:
            ledger.add(1)
        if system == pipeline.EXTRACT_SYSTEM:
            return dict(concepts=self.concepts[user])
        if system == pipeline.MERGE_SYSTEM:
            return dict(same=user in self.same)
        if system == revision.CONFLICT_SYSTEM:
            return dict(keep=self.keep)
        target = user.split('"')[1]
        return dict(prereqs=[dict(name=p, reason="needed") for p in self.parents.get(target, [])])


def install(monkeypatch, muse):
    monkeypatch.setattr(pipeline, "ask_json", muse)
    monkeypatch.setattr(revision, "ask_json", muse)


def old_graph():
    dag = PrereqDAG()
    dag.add_node("a", "a", ["o:q1"], ["o"])
    dag.add_node("b", "b", ["o:q2"], ["o"])
    dag.add_node("c", "c", ["o:q3"], ["o"])
    dag.add_edge("a", "b", "needed")
    dag.add_edge("b", "c", "needed")
    return dag


def new_q(number, text):
    return dict(id=f"n:q{number}", number=number, question=text, answers=[], text=text)


def test_concept_surviving_a_modified_question_keeps_edges_and_is_not_retired(monkeypatch):
    install(monkeypatch, FakeMuse(concepts={"q2 new": ["b"]}))
    diff = dict(unchanged=[("o:q1", "n:q1"), ("o:q3", "n:q3")], modified=[("o:q2", "n:q2")], removed=[], added=[])
    revised, report = revise(old_graph(), diff, [new_q(2, "q2 new")], "n", Ledger())
    assert revised.g.has_edge("a", "b") and revised.g.has_edge("b", "c")
    assert report["retired"] == [] and report["anomalies"]["broken_path_bridged"] == 0
    assert revised.g.nodes["b"]["provenance"] == {"n:q2"}
    assert revised.g.nodes["b"]["valid_versions"] == {"o", "n"}


def test_renamed_concept_is_absorbed_and_keeps_edges(monkeypatch):
    muse = FakeMuse(concepts={"q2 new": ["totally different"]}, same=['A: "b"\nB: "totally different"'])
    install(monkeypatch, muse)
    diff = dict(unchanged=[("o:q1", "n:q1"), ("o:q3", "n:q3")], modified=[("o:q2", "n:q2")], removed=[], added=[])
    revised, report = revise(old_graph(), diff, [new_q(2, "q2 new")], "n", Ledger())
    assert revised.g.has_edge("a", "b") and "totally_different" not in revised.g
    assert report["absorbed"] == {"totally_different": "b"}
    assert report["anomalies"]["ambiguous_merge"] == 1 and report["retired"] == []


def test_removed_concept_goes_on_record_with_its_versions(monkeypatch):
    install(monkeypatch, FakeMuse())
    diff = dict(unchanged=[("o:q1", "n:q1"), ("o:q3", "n:q3")], modified=[], removed=["o:q2"], added=[])
    revised, report = revise(old_graph(), diff, [], "n", Ledger())
    assert report["retired"] == ["b"] and revised.g.has_edge("a", "c")
    record = PrereqDAG.from_json(revised.to_json()).retired["b"]
    assert record["provenance"] == ["o:q2"] and record["valid_versions"] == ["o"] and record["retired_in"] == "n"


def test_new_concept_can_become_prereq_of_touched_node_but_old_nodes_do_not_rewire(monkeypatch):
    muse = FakeMuse(concepts={"q4": ["fresh", "c"]}, parents={"c": ["fresh", "a"]})
    install(monkeypatch, muse)
    diff = dict(unchanged=[("o:q1", "n:q1"), ("o:q2", "n:q2"), ("o:q3", "n:q3")], modified=[], removed=[],
                added=["n:q4"])
    revised, _ = revise(old_graph(), diff, [new_q(4, "q4")], "n", Ledger())
    assert revised.g.has_edge("fresh", "c")
    assert not revised.g.has_edge("a", "c")


def touched_b_reversing_b_to_c(keep):
    """A new question touches b, and its parent picks say c comes before b: the reverse of b -> c."""
    return FakeMuse(concepts={"q4": ["b"]}, parents={"b": ["c"]}, keep=keep)


TOUCH_B = dict(unchanged=[("o:q1", "n:q1"), ("o:q2", "n:q2"), ("o:q3", "n:q3")], modified=[], removed=[],
               added=["n:q4"])


def test_reversing_a_retained_edge_is_a_conflict_and_the_current_edge_can_win(monkeypatch):
    install(monkeypatch, touched_b_reversing_b_to_c("current"))
    revised, report = revise(old_graph(), TOUCH_B, [new_q(4, "q4")], "n", Ledger())
    assert report["anomalies"]["conflict"] == 1 and report["anomalies"]["cycle"] == 0
    assert revised.g.has_edge("b", "c") and not revised.g.has_edge("c", "b")


def test_reversing_a_retained_edge_can_replace_it_when_judged_right(monkeypatch):
    install(monkeypatch, touched_b_reversing_b_to_c("proposed"))
    revised, report = revise(old_graph(), TOUCH_B, [new_q(4, "q4")], "n", Ledger())
    assert report["anomalies"]["conflict"] == 1
    assert revised.g.has_edge("c", "b") and not revised.g.has_edge("b", "c")


def test_revision_keeps_every_parent_and_bridge_past_three(monkeypatch):
    """A retired node with four parents leaves four bridges on its child, and nothing trims them."""
    monkeypatch.setattr(pipeline, "ask_json", lambda *a, **k: {"prereqs": [], "concepts": []})
    dag = dag_of(*((p, "gone") for p in "abcd"), ("gone", "t"))
    dag.g.nodes["gone"]["provenance"] = {"old:removed"}
    diff = dict(removed=["old:removed"], added=[], modified=[], unchanged=[(f"old:{n}", f"new:{n}") for n in ["a", "b", "c", "d", "t"]])
    revised, report = revise(dag, diff, [], "new", Ledger())
    assert report["retired"] == ["gone"]
    assert sorted(revised.g.predecessors("t")) == ["a", "b", "c", "d"]


def test_learner_report_uses_the_given_version_and_rates_violations():
    old = dag_of(("a", "b"))
    new = PrereqDAG()
    new.add_node("x", provenance=["n:q1"], valid_versions=["n"])
    new.add_node("y", provenance=["o:q9"], valid_versions=["o"])
    ref = PrereqDAG()
    ref.add_node("y", provenance=["n:q2"])
    ref.add_node("x", provenance=["n:q1"])
    ref.add_edge("y", "x", "needed")
    report = revision.learner_report(old, new, ref.g, "n")
    assert report["stale_in_remaining"] == 1
    assert report["reference_orderings"] == 1 and report["violations_vs_reference"] == 1
    assert report["violation_rate"] == 1.0


def test_parse_ham_keeps_only_the_correct_answer():
    text = ("SUBELEMENT T1\nT1A01 (C) [97.1]\nWhich part of the rules?\nA. One\nB. Two\nC. Three\nD. Four\n~~\n"
            "T1A02 (A) [97.3] Inline question?\nA. Yes\nB. No\nC. Maybe\nD. Never\n~~\nT1B Group heading\n")
    parsed = parse_ham(text, "2022")
    assert [p["id"] for p in parsed] == ["2022:T1A01", "2022:T1A02"]
    assert parsed[0]["text"] == "Which part of the rules? Answers: Three"
    assert parsed[1]["question"] == "Inline question?" and parsed[1]["answers"] == ["Yes"]


def test_ham_pools_stay_locked_without_the_test_unlock(monkeypatch):
    monkeypatch.delenv("PREREQ_RUN_TEST", raising=False)
    with pytest.raises(PermissionError):
        load_ham()


def test_docx_text_reads_paragraphs_and_skips_tab_elements(tmp_path):
    path = tmp_path / "pool.docx"
    xml = ('<w:document><w:p><w:r><w:t>T1A01 </w:t></w:r><w:r><w:tab/><w:t xml:space="preserve">(C) &amp; more</w:t>'
           '</w:r></w:p><w:p><w:r><w:t>Next</w:t></w:r></w:p></w:document>')
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)
    assert docx_text(path).split("\n")[:2] == ["T1A01 (C) & more", "Next"]
