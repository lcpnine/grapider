import json

import networkx as nx
import numpy as np
import pytest

from prereq import experiment as ex
from prereq import llm
from prereq.materials import Library, name_pattern, normal


class FakeDev:
    names = {i: i.upper() for i in "abcde"}
    ids = {i.upper(): i for i in "abcde"}
    positive = {("a", "b"), ("b", "c")}
    negative = {("a", "d"), ("d", "e"), ("c", "a")}
    reach = {("a", "b"), ("b", "c"), ("a", "c")}
    universe = reach | negative


def test_name_pattern_matches_plural_but_not_inside_a_word():
    pattern = name_pattern("Recurrent neural networks")
    assert pattern.search(normal("Recurrent Neural Network models"))
    assert pattern.search(normal("recurrent-neural networks"))
    assert not pattern.search(normal("recurrent neural"))
    assert not name_pattern("parsing").search(normal("reparsing the text"))


def test_library_limits_pages_per_document_and_ranks_title_hits_first():
    pages = [dict(doc="1", title="Other", page=i, text=f"page {i} mentions entropy here and more text") for i in range(5)]
    pages.append(dict(doc="2", title="Entropy basics", page=0, text="Entropy is a measure of uncertainty in a variable"))
    picked = Library(pages).passages("Entropy")
    assert picked[0]["title"] == "Entropy basics"
    assert sum(p["title"] == "Other" for p in picked) == 2


def test_placebo_never_uses_a_concept_that_mentions_the_target():
    real = {
        "alpha": [dict(title="t", text="alpha is here and also gamma " * 3)],
        "beta": [dict(title="t", text="beta explained at some length " * 3)],
        "gamma": [dict(title="t", text="gamma explained and alpha noted " * 3)],
        "empty": [],
    }
    placebo = ex.placebo_passages(real)
    assert set(placebo) == {"alpha", "beta", "gamma"}
    assert "beta" in placebo["alpha"][0]["text"]
    for target, passages in placebo.items():
        assert target not in " ".join(p["text"] for p in passages)
        assert sum(len(p["text"]) for p in passages) <= sum(len(p["text"]) for p in real[target])


def test_direct_counts_score_labeled_picks_only():
    ids = set("abcde")
    pos = ex.scope_pairs(FakeDev.positive, ids)
    labeled = pos | ex.scope_pairs(FakeDev.negative, ids)
    picks = {("a", "b"), ("a", "d"), ("e", "a")}
    result = ex.prf(ex.per_target(picks, pos, labeled, sorted(ids)))
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["scored"] == 2


def test_scope_excludes_pairs_crossing_the_split():
    assert ex.scope_pairs({("a", "b"), ("a", "x")}, {"a", "b"}) == {("a", "b")}


def test_paired_gain_is_zero_for_identical_arms():
    counts = np.array([[1, 2, 2], [0, 1, 1], [2, 2, 3]], dtype=float)
    gain = ex.paired_gain(counts, counts)
    assert gain["f1"]["gain"] == 0
    assert gain["f1"]["low"] == 0 and gain["f1"]["high"] == 0


def test_bits_closure_matches_networkx_and_scores_removals():
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("a", "d"), ("d", "e")])
    bits = ex.Bits(graph, FakeDev, set("abcde"))
    assert bits.pairs() == len(ex.closure_of(graph))
    assert bits.score() == (3, 2)
    assert bits.score({("a", "b")}) == (1, 2)
    removal = dict((e, gain) for gain, e in bits.impact())
    assert removal[("a", "b")] == 2
    assert removal[("b", "c")] == 2


def test_oracle_removes_only_edges_gold_calls_wrong():
    graph = nx.DiGraph([("a", "b"), ("b", "c"), ("a", "d"), ("d", "e")])
    bits = ex.Bits(graph, FakeDev, set("abcde"))
    curve = ex.oracle_curve(bits, FakeDev)
    assert curve["all wrong"] == {("a", "d"), ("d", "e")}


def test_holdout_refuses_before_decisions_and_after_one_scoring(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "FREEZE", tmp_path / "freeze.json")
    monkeypatch.setattr(ex, "LOG", tmp_path / "log.jsonl")
    with pytest.raises(FileNotFoundError):
        ex.holdout()
    (tmp_path / "freeze.json").write_text("{}")
    (tmp_path / "log.jsonl").write_text(json.dumps(dict(config=dict(stage="holdout"))) + "\n")
    with pytest.raises(RuntimeError):
        ex.holdout()


def test_uptake_row_separates_parents_the_excerpts_name():
    pos = {("a", "t"), ("b", "t")}
    labeled = pos | {("c", "t"), ("d", "t")}
    named = {("a", "t"), ("c", "t")}
    row = ex.uptake_row({("a", "t"), ("c", "t"), ("d", "t")}, pos, labeled, named)
    assert row["share_named"] == 2 / 3
    assert row["named"] == dict(scored=2, precision=0.5)
    assert row["unnamed"] == dict(scored=1, precision=0.0)
    assert row["recall_named"] == 1.0 and row["recall_unnamed"] == 0.0


def test_change_row_scores_kept_dropped_and_added_edges():
    pos = {("a", "t")}
    labeled = pos | {("b", "t"), ("c", "t")}
    row = ex.change_row({("a", "t"), ("b", "t")}, {("a", "t"), ("c", "t")}, pos, labeled)
    assert row["kept"]["precision"] == 1.0
    assert row["dropped"]["precision"] == 0.0
    assert row["added"] == dict(scored=1, precision=0.0)


def test_dispute_groups_use_only_pairs_labeled_by_both_datasets():
    row = lambda a, b, source, label: dict(prereq_id=a, dependent_id=b, source=source, label=label)
    groups = ex.dispute_groups([
        row("a", "b", "lb1", 1), row("a", "b", "lb2", 0),
        row("b", "c", "lb1", 1), row("b", "c", "lb2", 1),
        row("c", "d", "lb1", 0), row("c", "d", "lb2", 0),
        row("d", "e", "lb1", 1),
    ])
    assert [p for p, _ in groups["disputed"]] == [("a", "b")]
    assert [p for p, _ in groups["both_yes"]] == [("b", "c")]
    assert [p for p, _ in groups["both_no"]] == [("c", "d")]


def test_tally_arm_keeps_unrated_out_of_precision_but_in_the_floor():
    ratings = dict(yes=[1, 2], unclear=[3], no=[4])
    row = ex.tally_arm([1, 2, 3, 4, 5, 6, 7, 8], ratings)
    assert (row["yes"], row["unclear"], row["no"], row["unrated"]) == (2, 1, 1, 4)
    assert row["strict"] == 0.5 and row["lenient"] == 0.75 and row["floor"] == 0.25


def test_wilson_stays_inside_zero_and_one():
    low, high = ex.wilson(0, 40)
    assert low == pytest.approx(0, abs=1e-9) and 0 < high < 0.1
    low, high = ex.wilson(20, 40)
    assert low < 0.5 < high


class Fake:
    calls = []

    def __init__(self, **_):
        from types import SimpleNamespace
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        from types import SimpleNamespace
        Fake.calls.append(kwargs["seed"])
        message = SimpleNamespace(content=json.dumps(dict(n=len(Fake.calls))))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=1))


@pytest.fixture
def caches(monkeypatch, tmp_path):
    Fake.calls = []
    monkeypatch.setattr(llm, "CACHE", tmp_path / "public.jsonl")
    monkeypatch.setattr(llm, "PRIVATE_CACHE", tmp_path / "private" / "cache.jsonl")
    monkeypatch.setattr(llm, "_cache", None)
    monkeypatch.setattr(llm, "OpenAI", Fake)
    monkeypatch.setenv("META_API_KEY", "unused")
    return tmp_path


def test_replicates_get_their_own_key_and_seed(caches):
    first = llm.ask_json("s", "u")
    again = llm.ask_json("s", "u")
    second = llm.ask_json("s", "u", replicate=1)
    assert first == again != second
    assert Fake.calls == [0, 1]


def test_private_answers_go_to_the_private_file_and_are_reloaded(caches, monkeypatch):
    first = llm.ask_json("s", "slide text", private=True)
    assert not (caches / "public.jsonl").exists()
    assert (caches / "private" / "cache.jsonl").exists()
    monkeypatch.setattr(llm, "_cache", None)
    assert llm.ask_json("s", "slide text", private=True) == first
    assert len(Fake.calls) == 1


def test_capped_replicate_zero_keeps_the_prompt_already_cached():
    assert ex.ARMS["capped"]["system"].startswith("You build a prerequisite graph.")
    assert "up to 3" in ex.ARMS["capped"]["system"]
    assert "up to" not in ex.UNCAPPED
    assert ex.SOURCE.startswith(ex.UNCAPPED)


def test_prompt_text_matches_what_was_logged_and_cached():
    """A changed hash would silently change a baseline and miss its cached answers."""
    assert {arm: ex.hash_text(config["system"]) for arm, config in ex.ARMS.items()} == {
        "capped": "a6ffb8c5fc47", "uncapped": "947202d3f6c8", "source": "fe201ea19344", "placebo": "fe201ea19344"}
    assert ex.ARMS["capped"]["cap"] == 3 and ex.ARMS["uncapped"]["cap"] is None
    assert {name: ex.hash_text(ex.EDGE_PROMPTS[name]) for name in ("e1_direct", "e2_strict", "e3_chain", "e4_uncapped")} == {
        "e1_direct": "f8de7ec0723a", "e2_strict": "a6ffb8c5fc47", "e3_chain": "8b70a769db1b", "e4_uncapped": "947202d3f6c8"}


def write_sheet(path, rows):
    path.write_text("item,A,B,rater_1_A_needed_for_B,rater_2_A_needed_for_B\n" + "\n".join(rows) + "\n")


def test_read_raters_normalizes_answers_and_skips_blank_columns(tmp_path):
    sheet = tmp_path / "s.csv"
    write_sheet(sheet, ['0,"a, b",c,Yes,', "1,d,e,unsure,", "2,f,g,,"])
    assert ex.read_raters(sheet) == {"rater_1_A_needed_for_B": {0: "yes", 1: "unclear"}}


def test_read_raters_rejects_an_unknown_answer(tmp_path):
    sheet = tmp_path / "s.csv"
    write_sheet(sheet, ["0,a,b,maybe,"])
    with pytest.raises(ValueError, match="item 0"):
        ex.read_raters(sheet)


def test_rater_agreement_kappa_uses_items_both_decided():
    first = {0: "yes", 1: "yes", 2: "no", 3: "no", 4: "unclear", 5: "yes"}
    second = {0: "yes", 1: "no", 2: "no", 3: "no", 4: "yes"}
    out = ex.rater_agreement(first, second)
    assert out["both_rated"] == 5 and out["both_decided"] == 4
    assert out["raw_agreement"] == pytest.approx(3 / 5)
    assert out["kappa"] == pytest.approx(.5)


def test_rater_agreement_has_no_kappa_without_decided_items():
    assert ex.rater_agreement({0: "unclear"}, {0: "yes"})["kappa"] is None


def test_human_reviews_stay_none_until_a_column_is_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "RUNS", tmp_path)
    write_sheet(tmp_path / "review_packet.csv", ["0,a,b,,", "1,c,d,,"])
    write_sheet(tmp_path / "kappa_review.csv", ["0,a,b,,"])
    packet_key, kappa_key = {"capped": [0], "uncapped": [1]}, {"0": {"lb1": 1, "lb2": 0}}
    assert ex.human_reviews(packet_key, kappa_key) is None
    write_sheet(tmp_path / "review_packet.csv", ["0,a,b,yes,no", "1,c,d,no,no"])
    write_sheet(tmp_path / "kappa_review.csv", ["0,a,b,no,"])
    out = ex.human_reviews(packet_key, kappa_key)
    assert out["packet_arms"]["rater_1_A_needed_for_B"]["capped"]["yes"] == 1
    assert out["packet_arms"]["rater_2_A_needed_for_B"]["capped"]["no"] == 1
    assert out["packet_agreement"]["both_rated"] == 2
    assert out["kappa_sheet"]["rater_1_A_needed_for_B"] == {"lb1_only": {"no": 1}}
    assert out["kappa_agreement"] is None
