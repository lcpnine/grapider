"""Freeze judge_dev and system_dev to data/splits. Refuses to overwrite a frozen split."""
import hashlib
import json
import random

from .schema import ROOT, load_lb1, load_lb2

SEED = 20261030
JUDGE_FRACTION = 0.2
LB_ROOT = ROOT / "data" / "raw" / "lecturebank" / "LB-Paper"
SPLIT_DIR = ROOT / "data" / "splits"


def make_splits(lb1, lb2, seed=SEED):
    """Split by concept. A judge_dev concept is removed from LB2 too, so it cannot leak."""
    c1, e1, _ = lb1
    c2, e2, _ = lb2
    ids = sorted(c["id"] for c in c1)
    judge = set(random.Random(seed).sample(ids, round(len(ids) * JUDGE_FRACTION)))
    in_judge = lambda e: e["prereq_id"] in judge and e["dependent_id"] in judge
    in_system = lambda e: e["prereq_id"] not in judge and e["dependent_id"] not in judge
    all_edges = e1 + e2
    judge_dev = dict(concepts=[c for c in c1 if c["id"] in judge], edges=[e for e in e1 if in_judge(e)])
    seen = {}
    for c in c1 + c2:
        if c["id"] not in judge:
            seen.setdefault(c["id"], c)
    system_dev = dict(concepts=list(seen.values()), edges=[e for e in all_edges if in_system(e)])
    report = dict(
        seed=seed,
        crossing_pairs_dropped=len(all_edges) - len(system_dev["edges"]) - len(judge_dev["edges"]),
        lb2_concepts_removed_as_judge=len({c["id"] for c in c2} & judge),
    )
    return judge_dev, system_dev, report


def freeze():
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = SPLIT_DIR / "manifest.json"
    if manifest.exists():
        raise FileExistsError(f"{manifest} exists; splits are frozen")
    lb1, lb2 = load_lb1(LB_ROOT), load_lb2(LB_ROOT)
    judge_dev, system_dev, report = make_splits(lb1, lb2)
    report["normalize"] = dict(lb1=lb1[2], lb2=lb2[2])
    for name, split in (("judge_dev", judge_dev), ("system_dev", system_dev)):
        blob = json.dumps(split, indent=0, sort_keys=True).encode()
        (SPLIT_DIR / f"{name}.json").write_bytes(blob)
        report[name] = dict(
            sha256=hashlib.sha256(blob).hexdigest(),
            concepts=len(split["concepts"]),
            pairs=len(split["edges"]),
            positives=sum(e["label"] for e in split["edges"]),
        )
    manifest.write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=1))
