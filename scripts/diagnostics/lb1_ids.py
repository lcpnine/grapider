"""Is LB1's annotation id space aligned with 208topics.csv? Dev labels only, no model calls."""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from prereq.dag import MAX_PARENTS
from prereq.llm import CACHE, MUSE_MODEL
from prereq.pipeline import EDGE_PROMPTS
from prereq.schema import canon
from prereq.splits import SPLIT_DIR

ROOT = Path(__file__).resolve().parents[2]
LB = ROOT / "data/raw/lecturebank/LB-Paper"
topics = [r for r in csv.reader(open(LB / "208topics.csv", encoding="utf8", errors="replace")) if r]
by_id = {int(r[0]): r[1] for r in topics}
ann = [(int(a), int(b), int(c)) for a, b, c in csv.reader(open(LB / "prerequisite_annotation.csv", encoding="utf8", errors="replace"))]
used = {x for a, b, _ in ann for x in (a, b)}
print("topic ids", min(by_id), max(by_id), "gaps", sorted(set(range(min(by_id), max(by_id) + 1)) - set(by_id)))
print("annotation ids", min(used), max(used), "gaps", sorted(set(range(min(used), max(used) + 1)) - used))
print("topics never annotated", {i: by_id[i] for i in sorted(set(by_id) - used)})
print("annotated, no topic", sorted(used - set(by_id)))
print("last 12 topics", [(i, by_id[i]) for i in sorted(by_id)[-12:]])
out, inn = Counter(), Counter()
for a, b, c in ann:
    if c:
        out[a] += 1
        inn[b] += 1
print("largest raw out-degree", [(i, by_id.get(i, "?"), n) for i, n in out.most_common(14)])
print("children of 210", [by_id.get(b) for a, b, c in ann if a == 210 and c][:25], sum(1 for a, b, c in ann if a == 210 and c))
print("children of 211", [by_id.get(b) for a, b, c in ann if a == 211 and c][:25], sum(1 for a, b, c in ann if a == 211 and c))
for i in (205, 206, 207, 208, 209):
    print(i, by_id.get(i), "out", out[i], "in", inn[i], "children", [by_id.get(b) for a, b, c in ann if a == i and c][:12])

# Annotator block order: pairs are written in id order, so show which ids were annotated as parents with which others
rows_by_a = Counter(a for a, _, _ in ann)
print("rows per parent id, ends", [(i, rows_by_a[i]) for i in sorted(rows_by_a)[:5]], [(i, rows_by_a[i]) for i in sorted(rows_by_a)[-8:]])

# Shift test: do the model's cached picks hit LB1 labels better under another id mapping?
split = json.loads((SPLIT_DIR / "system_dev.json").read_text())
names = {c["id"]: c["name"] for c in split["concepts"]}
ids = {n: i for i, n in names.items()}
cache = {}
for row in map(json.loads, CACHE.open()):
    cache.setdefault(row["key"], row)
picks = set()
for target in ids:
    others = sorted(n for n in ids if n != target)
    user = f'Target: "{target}"\nConcepts:\n' + "\n".join(f"- {n}" for n in others)
    key = hashlib.sha256(json.dumps([MUSE_MODEL, EDGE_PROMPTS["e2_strict"], user]).encode()).hexdigest()
    known = {canon(n): n for n in others}
    chosen = []
    for p in cache[key]["out"].get("prereqs", []):
        n = known.get(canon(str(p.get("name", ""))))
        if n and n not in chosen:
            chosen.append(n)
    picks |= {(ids[n], ids[target]) for n in chosen[:MAX_PARENTS]}
raw_id_of = defaultdict(list)
for i, n in by_id.items():
    raw_id_of[canon(n)].append(i)

print("\nmodel hits by raw id band, pairs with a positive LB1 label under the file's mapping")
band = Counter()
hits = Counter()
for a, b, c in ann:
    if not c or a not in by_id or b not in by_id:
        continue
    k = (a // 40) * 40
    band[k] += 1
    hits[k] += (canon(by_id[a]), canon(by_id[b])) in picks
print({k: f"{hits[k]}/{band[k]}" for k in sorted(band)})

for shift in (-3, -2, -1, 0, 1, 2, 3):
    pos = {(by_id.get(a + shift), by_id.get(b + shift)) for a, b, c in ann if c}
    pos = {(canon(x), canon(y)) for x, y in pos if x and y}
    print(f"shift {shift:+d}: positives {len(pos)}, model picks that hit {len(pos & picks)}")

print("\nparent-side shift only, for the top raw parents: which topic's name do the model's own parents suggest?")
for i in (208, 209, 210, 211):
    kids = {by_id[b] for a, b, c in ann if a == i and c and b in by_id}
    votes = Counter()
    for k in kids:
        for p, t in picks:
            if t == canon(k):
                votes[names[p]] += 1
    print(i, by_id.get(i), "children", len(kids), "model parents of those children", votes.most_common(8))

for f in ("LB-Paper/README.md", "LB-Paper/vocabulary.txt"):
    r = subprocess.run(["git", "-C", str(ROOT / "data/raw/lecturebank"), "show", f"HEAD:{f}"], capture_output=True, text=True, timeout=120)
    print(f"\n== {f} rc {r.returncode}\n", r.stdout[:1800], r.stderr[:200])
