"""Parse question pools and diff two versions. No LLM in this module."""
import difflib
import re
import subprocess

from .schema import ROOT

CIVICS = ROOT / "data" / "revision" / "civics"
MODIFIED_THRESHOLD = 0.75


def parse_questions(text, version):
    """Return [{id, number, question, answers, text}] from numbered question text."""
    questions, current = [], None
    for raw in text.replace("\r", "").split("\n"):
        raw = re.sub(r"\s*\d+ of \d+\s+uscis\.gov/citizenship\s*$", "", raw)
        line = raw.strip()
        m = re.match(r"^(\d{1,3})\.\s+(.+)$", line)
        if m and (current is None or int(m[1]) == current["number"] + 1):
            current = dict(number=int(m[1]), question=m[2], answers=[])
            questions.append(current)
        elif current is not None and re.match(r"^[.•▪]\s*", line):
            current["answers"].append(re.sub(r"^[.•▪]\s*", "", line))
        elif current is not None and line and current["answers"] and raw.startswith((" " * 6, "\t")):
            current["answers"][-1] += " " + line
        elif current is not None and line and not current["answers"] and not line.isupper() and "?" not in current["question"]:
            current["question"] += " " + line
    for q in questions:
        q["question"] = re.sub(r"\s*\*\s*$", "", q["question"])
        q["id"] = f"{version}:q{q['number']}"
        q["text"] = q["question"] + " Answers: " + "; ".join(q["answers"])
    return questions


def load_civics():
    old = open(CIVICS / "civics_2008.txt", encoding="utf8", errors="replace").read()
    new = subprocess.run(["pdftotext", "-layout", str(CIVICS / "civics_2025.pdf"), "-"],
                         capture_output=True, text=True, check=True).stdout
    return parse_questions(old, "2008"), parse_questions(new, "2025")


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


def diff_questions(old, new):
    """Match by question wording. Same wording and answers is unchanged; otherwise modified."""
    free = {q["id"]: q for q in new}
    unchanged, modified, removed = [], [], []
    candidates = [(difflib.SequenceMatcher(None, _norm(a["question"]), _norm(b["question"])).ratio(), a["id"], b["id"])
                  for a in old for b in new]
    by_id = {q["id"]: q for q in old}
    matched = {}
    for score, a, b in sorted(candidates, reverse=True):
        if score < MODIFIED_THRESHOLD:
            break
        if a not in matched and b in free:
            matched[a] = b
            del free[b]
    new_by_id = {q["id"]: q for q in new}
    for a, q in by_id.items():
        if a not in matched:
            removed.append(a)
            continue
        same = _norm(q["text"]) == _norm(new_by_id[matched[a]]["text"])
        (unchanged if same else modified).append((a, matched[a]))
    return dict(unchanged=unchanged, modified=modified, removed=removed, added=sorted(free))
