"""Parse question pools and diff two versions. No LLM in this module."""
import difflib
import html
import re
import subprocess
import zipfile

from .schema import ROOT, TEST_DIR, guard_test_path

CIVICS = ROOT / "data" / "revision" / "civics"
HAM = TEST_DIR / "raw" / "ham"
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


def docx_text(path):
    """Paragraph text of a .docx, one paragraph per line."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf8")
    return "\n".join(html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p)))
                     for p in xml.split("</w:p>"))


def parse_ham(text, version):
    """Parse NCVEC pool text: 'T1A01 (C) [97.1]', the question, options A. to D., then '~~'."""
    questions, current = [], None
    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        m = re.match(r"^([TGE]\d[A-Z]\d{2})\s*\(([A-D])\)(?:\s*\[[^\]]*\])?\s*(.*)$", line)
        option = re.match(r"^([A-D])\.\s*(.*)$", line)
        if m:
            current = dict(number=m[1], key=m[2], question=m[3], answers=[])
            questions.append(current)
        elif current is None or line.startswith("~~"):
            current = None
        elif option:
            if option[1] == current["key"]:
                current["answers"].append(option[2])
        elif line and not current["answers"]:
            current["question"] = (current["question"] + " " + line).strip()
    for q in questions:
        q["id"] = f"{version}:{q['number']}"
        q["text"] = q["question"] + " Answers: " + "; ".join(q["answers"])
    return questions


def load_ham():
    old, new = HAM / "tech_2022_2026.docx", HAM / "tech_2026_2030.docx"
    guard_test_path(old)
    guard_test_path(new)
    return parse_ham(docx_text(old), "2022"), parse_ham(docx_text(new), "2026")


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
