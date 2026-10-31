"""Cached Muse calls with token accounting. Every call is keyed by model, system and user text."""
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from openai import OpenAI

from .schema import ROOT

load_dotenv(ROOT / ".env", override=True)

MUSE_MODEL = "muse-spark-1.3"
RUNS = ROOT / "runs"
CACHE = RUNS / "cache_muse.jsonl"
PRIVATE_CACHE = RUNS / "private" / "cache_muse.jsonl"
_lock = threading.Lock()
_cache = None


def _load():
    """Load once, publishing only the full cache. The first answer for a key is the answer of record."""
    global _cache
    with _lock:
        if _cache is None:
            cache = {}
            for path in (CACHE, PRIVATE_CACHE):
                if path.exists():
                    for row in map(json.loads, path.open()):
                        cache.setdefault(row["key"], row)
            _cache = cache
    return _cache


class Ledger:
    """Counts what a workflow would cost, whether or not the answer came from the cache."""

    def __init__(self):
        self.calls = self.tokens = 0

    def add(self, tokens):
        with _lock:
            self.calls += 1
            self.tokens += tokens

    def to_dict(self):
        return dict(llm_calls=self.calls, llm_tokens=self.tokens)


def ask_json(system, user, ledger=None, model=MUSE_MODEL, replicate=0, private=False):
    """replicate > 0 asks again under its own cache key and seed. private keeps the answer out of git."""
    parts = [model, system, user] + ([replicate] if replicate else [])
    key = hashlib.sha256(json.dumps(parts).encode()).hexdigest()
    row = _load().get(key)
    if row is None:
        client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")
        for attempt in range(6):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, seed=replicate, response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                out = json.loads(r.choices[0].message.content)
                break
            except Exception as err:
                if attempt == 5:
                    raise
                print(f"retry after {type(err).__name__}: {str(err)[:120]}", file=sys.stderr)
                time.sleep(2**attempt)
        row = dict(key=key, out=out, tokens=r.usage.total_tokens)
        with _lock:
            if key in _cache:
                row = _cache[key]
            else:
                _cache[key] = row
                path = PRIVATE_CACHE if private else CACHE
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
    if ledger:
        ledger.add(row["tokens"])
    return row["out"]


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(fn, items))


def log_run(name, config, metrics):
    RUNS.mkdir(exist_ok=True)
    row = dict(time=time.strftime("%FT%T"),
               config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
               model=config.get("model", MUSE_MODEL), prompt_version=config.get("prompt_version"),
               config=config, metrics=metrics)
    with (RUNS / f"{name}.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
