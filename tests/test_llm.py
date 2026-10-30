import json
from types import SimpleNamespace

import pytest

from prereq import llm


class FakeClient:
    calls = []

    def __init__(self, **_):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **_):
        FakeClient.calls.append(1)
        message = SimpleNamespace(content=json.dumps(dict(source="api")))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(total_tokens=1))


@pytest.fixture
def cache_file(monkeypatch, tmp_path):
    FakeClient.calls = []
    path = tmp_path / "cache.jsonl"
    monkeypatch.setattr(llm, "CACHE", path)
    monkeypatch.setattr(llm, "_cache", None)
    monkeypatch.setattr(llm, "OpenAI", FakeClient)
    monkeypatch.setenv("META_API_KEY", "unused")
    return path


def row(key, out):
    return json.dumps(dict(key=key, out=out, tokens=1))


def key_of(system, user):
    import hashlib
    return hashlib.sha256(json.dumps([llm.MUSE_MODEL, system, user]).encode()).hexdigest()


def test_threads_starting_together_never_see_a_half_loaded_cache(cache_file):
    users = [f"q{i}" for i in range(64)]
    filler = [row(f"filler{i}", {}) for i in range(5000)]
    cache_file.write_text("\n".join(filler + [row(key_of("s", u), dict(source="cache")) for u in users]) + "\n")
    answers = llm.pmap(lambda u: llm.ask_json("s", u), users, workers=16)
    assert FakeClient.calls == []
    assert all(a == dict(source="cache") for a in answers)


def test_first_answer_wins_when_a_key_was_written_twice(cache_file):
    cache_file.write_text(row(key_of("s", "u"), dict(answer="first")) + "\n" + row(key_of("s", "u"), dict(answer="second")) + "\n")
    assert llm.ask_json("s", "u") == dict(answer="first")


def test_identical_prompts_racing_share_one_answer_and_one_cache_row(cache_file):
    answers = llm.pmap(lambda _: llm.ask_json("s", "same"), range(8), workers=8)
    assert all(a == answers[0] for a in answers)
    assert len(cache_file.read_text().splitlines()) == 1
