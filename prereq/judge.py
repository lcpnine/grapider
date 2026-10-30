"""Prerequisite judge: a non-Muse classifier, tuned on judge_dev only.

Usage: python -m prereq.judge grid | python -m prereq.judge freeze
"""
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

from .schema import ROOT
from .splits import SPLIT_DIR

load_dotenv(ROOT / ".env", override=True)

RUNS = ROOT / "runs"
CACHE = RUNS / "cache_judge.jsonl"
FROZEN = ROOT / "data" / "judge_frozen.json"
SPLIT_SEED = 7
GRID_MODELS = ["gpt-6-luna", "claude-haiku-5-5"]

OUTPUT = 'Reply with JSON only: {"reason": "<one line>", "requires": true or false}.'
QUESTION = 'Concept A: "{a}"\nConcept B: "{b}"\nIs A a prerequisite of B?'
PROMPTS = {
    "v1_direct": "You judge prerequisite relations between study concepts. "
                 "A is a prerequisite of B if a learner should learn A before B. " + OUTPUT,
    "v2_strict": "You judge prerequisite relations between study concepts. Answer true only if a learner "
                 "cannot properly understand B without first knowing A. Being related, being in the same field, "
                 "or A being generally useful background is not enough. Most pairs are not prerequisites, and "
                 "the relation is directional: if B is needed for A, answer false. " + OUTPUT,
    "v3_examples": "You judge prerequisite relations between study concepts. Answer true only if A is a direct "
                   "or clear foundation that a course on B would assume. Most pairs are not prerequisites.\n"
                   "Examples:\nA=fractions, B=adding unlike fractions: true\n"
                   "A=adding unlike fractions, B=fractions: false (reversed)\n"
                   "A=photosynthesis, B=cellular respiration: false (related, neither is required)\n"
                   "A=cell membrane, B=osmosis in cells: true\n" + OUTPUT,
}


def load_pairs():
    return json.loads((SPLIT_DIR / "judge_dev.json").read_text())


def tune_holdout(edges):
    """Fixed stratified split of judge_dev pairs: choose on tune, report on holdout."""
    rng = random.Random(SPLIT_SEED)
    tune, holdout = [], []
    for label in (0, 1):
        group = [e for e in edges if e["label"] == label]
        rng.shuffle(group)
        tune += group[: len(group) // 2]
        holdout += group[len(group) // 2 :]
    return tune, holdout


def metrics(gold, pred):
    return dict(
        n=len(gold), positives=sum(gold), predicted_positive=sum(pred),
        accuracy=accuracy_score(gold, pred),
        precision=precision_score(gold, pred, zero_division=0),
        recall=recall_score(gold, pred, zero_division=0),
        f1=f1_score(gold, pred, zero_division=0),
        kappa=cohen_kappa_score(gold, pred),
    )


def bootstrap_ci(gold, pred, n=2000, seed=0):
    """95% percentile intervals for F1 and kappa."""
    rng = random.Random(seed)
    idx = range(len(gold))
    f1s, kappas = [], []
    for _ in range(n):
        s = rng.choices(idx, k=len(gold))
        g, p = [gold[i] for i in s], [pred[i] for i in s]
        f1s.append(f1_score(g, p, zero_division=0))
        kappas.append(cohen_kappa_score(g, p) if len(set(g) | set(p)) > 1 else 0.0)
    pick = lambda v: [sorted(v)[int(n * 0.025)], sorted(v)[int(n * 0.975)]]
    return dict(f1=pick(f1s), kappa=pick(kappas))


def _read_cache():
    if not CACHE.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, CACHE.open())}


def _key(model, prompt_id, a, b):
    return hashlib.sha256(json.dumps([model, PROMPTS[prompt_id], a, b]).encode()).hexdigest()


def _complete(clients, model, prompt_id, a, b):
    """Return the model's raw text. Neither model accepts temperature=0, so none is sent."""
    system, user = PROMPTS[prompt_id], QUESTION.format(a=a, b=b)
    if model.startswith("claude-"):
        r = clients["anthropic"].messages.create(
            model=model, max_tokens=2048, system=system, output_config={"effort": "low"},
            messages=[{"role": "user", "content": user}],
        )
        if r.stop_reason == "refusal":
            raise RuntimeError(f"refusal on {a!r} / {b!r}")
        return next(block.text for block in r.content if block.type == "text")
    r = clients["openai"].chat.completions.create(
        model=model, seed=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return r.choices[0].message.content


def _parse(text):
    """Accept a JSON object even when it is wrapped in a code fence or other text."""
    return json.loads(text[text.index("{") : text.rindex("}") + 1])


def _ask(clients, model, prompt_id, a, b):
    for attempt in range(6):
        try:
            out = _parse(_complete(clients, model, prompt_id, a, b))
            return bool(out["requires"]), str(out.get("reason", ""))
        except Exception as err:
            if attempt == 5:
                raise
            print(f"retry after {type(err).__name__}", file=sys.stderr)
            time.sleep(2**attempt)


def predict(model, prompt_id, pairs):
    """Return one 0/1 per (a, b) name pair. Answers are cached, so reruns cost nothing."""
    cache = _read_cache()
    todo = [p for p in dict.fromkeys(pairs) if _key(model, prompt_id, *p) not in cache]
    if todo:
        clients = dict(openai=OpenAI(), anthropic=anthropic.Anthropic())
        with ThreadPoolExecutor(16) as pool, CACHE.open("a") as f:
            for (a, b), (ans, why) in zip(todo, pool.map(lambda p: _ask(clients, model, prompt_id, *p), todo)):
                row = dict(key=_key(model, prompt_id, a, b), a=a, b=b, requires=ans, reason=why)
                cache[row["key"]] = row
                f.write(json.dumps(row) + "\n")
    return [int(cache[_key(model, prompt_id, *p)]["requires"]) for p in pairs]


def log_run(config, result):
    RUNS.mkdir(exist_ok=True)
    row = dict(time=time.strftime("%FT%T"), config_hash=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12],
               model=config["model"], prompt_version=config["prompt_version"], config=config, metrics=result)
    with (RUNS / "judge.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")


def _evaluate(model, prompt_id, edges, names):
    pairs = [(names[e["prereq_id"]], names[e["dependent_id"]]) for e in edges]
    return [e["label"] for e in edges], predict(model, prompt_id, pairs)


def grid():
    """Score each candidate on the tune half only. Holdout stays unused here."""
    data = load_pairs()
    names = {c["id"]: c["name"] for c in data["concepts"]}
    tune, _ = tune_holdout(data["edges"])
    results = []
    for model in GRID_MODELS:
        for prompt_id in PROMPTS:
            result = metrics(*_evaluate(model, prompt_id, tune, names))
            log_run(dict(stage="grid", split="judge_dev.tune", model=model, prompt_version=prompt_id, split_seed=SPLIT_SEED), result)
            results.append((model, prompt_id, result))
            print(model, prompt_id, {k: round(v, 3) for k, v in result.items()}, flush=True)
    return results


def freeze():
    """Pick the best kappa on tune, score holdout once, and write the frozen config."""
    if FROZEN.exists():
        raise FileExistsError(f"{FROZEN} exists; the judge is frozen")
    model, prompt_id, tuned = max(grid(), key=lambda r: r[2]["kappa"])
    data = load_pairs()
    names = {c["id"]: c["name"] for c in data["concepts"]}
    _, holdout = tune_holdout(data["edges"])
    gold, pred = _evaluate(model, prompt_id, holdout, names)
    result = metrics(gold, pred)
    result["ci95"] = bootstrap_ci(gold, pred)
    baseline = metrics(gold, [0] * len(gold))
    config = dict(stage="freeze", split="judge_dev.holdout", model=model, prompt_version=prompt_id, split_seed=SPLIT_SEED)
    log_run(config, result)
    FROZEN.write_text(json.dumps(dict(model=model, prompt_version=prompt_id, prompt=PROMPTS[prompt_id],
                                      tune=tuned, holdout=result, all_zero_baseline=baseline), indent=1))
    print("SELECTED", model, prompt_id)
    print("HOLDOUT", json.dumps(result))
    print("BASELINE", json.dumps(baseline))


if __name__ == "__main__":
    {"grid": grid, "freeze": freeze}[sys.argv[1]]()
