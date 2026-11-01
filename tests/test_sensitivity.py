import json

import numpy as np

from prereq import experiment as ex
from prereq import sensitivity as se
from prereq.materials import Library, normal


class FakeDev:
    names = {i: i for i in "abcd"}
    ids = names


def edge(a, b, label=1):
    return dict(prereq_id=a, dependent_id=b, label=label, source="lb1")


def test_view_marks_a_pair_positive_if_either_label_is_and_ignores_self_pairs():
    view = se.View(FakeDev.names, [edge("a", "b", 0), edge("a", "b", 1), edge("b", "c"), edge("c", "c"), edge("c", "d", 0)])
    assert view.positive == {("a", "b"), ("b", "c")}
    assert view.negative == {("c", "d")}
    assert ("a", "c") in view.reach
    assert ("c", "c") not in view.universe


def test_view_matches_dev_on_the_real_labels():
    dev = ex.Dev()
    view = se.View(dev.names, se.dev_edges())
    assert view.positive == dev.positive and view.negative == dev.negative
    assert view.reach == dev.reach and view.universe == dev.universe


def test_duplicate_rule_finds_the_known_pairs():
    assert se.same_concept("Seq2seq", "Sequence to sequence")
    assert se.same_concept("neural networks", "Artificial neural network")
    assert se.same_concept("T-SNE", "tsne")
    assert se.same_concept("DQN", "deep Q-network")
    assert se.same_concept("computation theory", "theory of computation")


def test_duplicate_rule_keeps_distinct_concepts_apart():
    assert not se.same_concept("Markov chains", "Markov Random Fields")
    assert not se.same_concept("information theory", "Information Theory: coding")
    assert not se.same_concept("Entropy", "Cross entropy")
    assert not se.same_concept("parsing", "cky parsing")
    assert not se.same_concept("A* search", "search")
    assert not se.same_concept("data structures", "data structures and algorithms")


def test_merge_never_moves_a_holdout_label_into_tune():
    tune = {"a", "b", "c"}
    pick_map, gold_map = se.merge_maps([("a", "b", "r"), ("c", "x", "r"), ("y", "z", "r")], tune)
    assert pick_map == {"a": "a", "b": "a", "c": "c", "x": "c"}
    assert gold_map == {"a": "a", "b": "a", "c": "c"}
    view = se.merged_view(FakeDev, [edge("x", "a"), edge("b", "c")], gold_map, pick_map)
    assert ("c", "a") not in view.positive
    assert view.positive == {("x", "a"), ("a", "c")}


def test_map_picks_drops_edges_inside_a_merged_concept():
    assert se.map_picks({("a", "b"): "w", ("b", "c"): "w"}, {"b": "a"}) == {("a", "c"): "w"}


def test_comention_ignores_a_name_only_inside_the_longer_name():
    pages = [dict(doc="1", title="t", page=0, text="cky parsing explained here"),
             dict(doc="2", title="t", page=0, text="cky parsing is one kind of parsing method"),
             dict(doc="3", title="t", page=0, text="entropy without the other one")]
    co = se.CoMention(Library(pages), {"p": "parsing", "c": "cky parsing", "e": "Entropy"})
    assert co.shared("p", "c") == {1}
    assert co.count(("p", "e")) == 0


def test_prune_ceiling_only_removes_wrong_edges_with_a_page():
    pos = {("a", "b"), ("b", "c")}
    labeled = pos | {("a", "c"), ("a", "d")}
    out = se.prune_ceiling([{("a", "b"), ("a", "c"), ("a", "d")}], pos, labeled, lambda e: e == ("a", "c"))
    assert np.isclose(out["direct_f1"], 2 * (1 / 3) * .5 / (1 / 3 + .5))
    assert np.isclose(out["ceiling_f1"], .5)
    assert out["wrong_edges_with_page"] == .5


def test_without_hubs_can_drop_positives_only_or_every_hub_pair(monkeypatch):
    monkeypatch.setattr(se, "HUBS", ("Bayes theorem",))

    class Dev:
        names = {"h": "Bayes theorem", "a": "a", "b": "b"}
        ids = {"Bayes theorem": "h", "a": "a", "b": "b"}

    edges = [edge("h", "a"), edge("h", "b", 0), edge("a", "b")]
    positives_gone = se.without_hubs(Dev, edges)
    assert positives_gone.positive == {("a", "b")} and ("h", "b") in positives_gone.negative
    every_pair_gone = se.without_hubs(Dev, edges, positives_only=False)
    assert every_pair_gone.positive == {("a", "b")} and not every_pair_gone.negative


def test_lift_interval_reports_the_rate_ratio():
    out = se.lift_interval([("a", "x"), ("b", "y")], [("c", "x"), ("d", "y"), ("e", "x"), ("f", "y")], lambda p: p[0] in "ac", {"x", "y"})
    assert np.isclose(out["rate_a"], .5) and np.isclose(out["rate_b"], .25) and np.isclose(out["lift"], 2)


def test_forbid_model_calls_stops_a_cache_miss(monkeypatch):
    from prereq import judge, llm
    monkeypatch.setattr(llm, "OpenAI", llm.OpenAI)
    monkeypatch.setattr(judge, "OpenAI", judge.OpenAI)
    monkeypatch.setattr(judge.anthropic, "Anthropic", judge.anthropic.Anthropic)
    se.forbid_model_calls()
    try:
        llm.OpenAI()
        raise AssertionError("should have refused")
    except RuntimeError:
        pass
