"""LectureBank slides as source material: fetch, extract, retrieve. No LLM in this module.

Slides and anything quoting them stay under gitignored paths. Dead links are expected.

Usage: python -m prereq.materials fetch | extract | coverage
"""
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

from .schema import ROOT
from .splits import SPLIT_DIR

CLONE = ROOT / "data" / "raw" / "lecturebank"
LIST_PATH = "LB-Paper/LectureBank1and2.tsv"
HOME = ROOT / "data" / "raw" / "lecturebank_materials"
FILES = HOME / "files"
TEXT = HOME / "text"
MANIFEST = HOME / "manifest.jsonl"
PAGES = HOME / "pages.jsonl"
MAX_BYTES = 60_000_000
TOP_PAGES = 6
PER_DOC = 2
PAGE_CHARS = 800
MIN_PAGE_CHARS = 40
TITLE_BONUS = 3


def lecture_list():
    out = subprocess.run(["git", "-C", str(CLONE), "show", f"HEAD:{LIST_PATH}"], capture_output=True, check=True)
    text = out.stdout.decode("utf8", errors="replace")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def direct_url(url):
    """Point GitHub and Dropbox page links at the file itself."""
    if "github.com" in url and "/blob/" in url:
        return url.replace("/blob/", "/raw/", 1)
    return url.replace("?dl=0", "?dl=1")


def file_kind(head):
    if head.startswith(b"%PDF"):
        return "pdf"
    return "zip" if head.startswith(b"PK") else None


def fetch_one(row):
    """Download one lecture, keeping it only if it really is a PDF or an Office zip."""
    result = dict(id=row["ID"], name=row["Name"], url=row["URL"], venue=row["Venue"], domain=row["Domain"])
    try:
        with requests.get(direct_url(row["URL"]), stream=True, timeout=(10, 30)) as r:
            result["status"] = r.status_code
            if r.status_code != 200:
                return result
            data = b""
            for chunk in r.iter_content(1 << 16):
                data += chunk
                if len(data) > MAX_BYTES:
                    result["error"] = "too large"
                    return result
    except requests.RequestException as err:
        return {**result, "status": 0, "error": type(err).__name__}
    kind = file_kind(data[:4])
    if kind is None:
        return {**result, "error": "not a document"}
    (FILES / f"{row['ID']}.{kind}").write_bytes(data)
    return {**result, "kind": kind, "bytes": len(data)}


def fetch():
    FILES.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in MANIFEST.open()} if MANIFEST.exists() else set()
    todo = [r for r in lecture_list() if r["ID"] not in done and r["URL"]]
    with ThreadPoolExecutor(12) as pool, MANIFEST.open("a") as f:
        for result in pool.map(fetch_one, todo):
            f.write(json.dumps(result) + "\n")
            f.flush()
    rows = [json.loads(line) for line in MANIFEST.open()]
    print("lectures", len(rows), "kept", sum("kind" in r for r in rows), Counter(r.get("error") or r.get("status") for r in rows).most_common(10))


def pdf_pages(path):
    out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, timeout=120)
    return out.stdout.decode("utf8", errors="replace").split("\f")


def pptx_pages(path):
    """Slide text from a .pptx, in slide order. Oversized zip members are skipped."""
    with zipfile.ZipFile(path) as z:
        slides = [i for i in z.infolist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", i.filename) and i.file_size < 20_000_000]
        slides.sort(key=lambda i: int(re.search(r"(\d+)\.xml$", i.filename).group(1)))
        return [" ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(i).decode("utf8", errors="replace"))) for i in slides]


def extract_one(row):
    path = FILES / f"{row['id']}.{row['kind']}"
    try:
        pages = pdf_pages(path) if row["kind"] == "pdf" else pptx_pages(path)
    except (subprocess.SubprocessError, zipfile.BadZipFile, OSError, KeyError) as err:
        return row, [], type(err).__name__
    return row, [" ".join(p.split()) for p in pages], None


def extract():
    rows = [r for r in map(json.loads, MANIFEST.open()) if "kind" in r]
    stats = Counter()
    with ThreadPoolExecutor(8) as pool, PAGES.open("w") as f:
        for row, pages, error in pool.map(extract_one, rows):
            kept = [(i, p) for i, p in enumerate(pages) if len(p) >= MIN_PAGE_CHARS]
            stats["documents"] += 1
            stats["with_text"] += bool(kept)
            stats["pages"] += len(kept)
            stats[f"error_{error}"] += bool(error)
            for i, page in kept:
                f.write(json.dumps(dict(doc=row["id"], title=row["name"], page=i, text=page)) + "\n")
    print(dict(stats))


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def name_pattern(name):
    """Match a concept name in normalized text, allowing a plural on the last word."""
    words = tokens(name)
    if len(words) > 1 and words[-1] == "s":
        words.pop()
    if not words:
        return None
    last = words[-1]
    if last.endswith("ies") and len(last) > 4:
        last = last[:-3] + "(?:y|ies)"
    elif last.endswith("s") and not last.endswith(("ss", "is", "us")) and len(last) > 3:
        last = last[:-1] + "(?:s|es)?"
    else:
        last = re.escape(last) + "(?:s|es)?"
    return re.compile(r"\b" + " ".join(map(re.escape, words[:-1])) + (" " if words[:-1] else "") + last + r"\b")


def normal(text):
    return " ".join(tokens(text))


class Library:
    """Pages indexed by word. A concept's passages are its best pages, at most PER_DOC per document."""

    def __init__(self, pages):
        self.pages = pages
        self.norm = [normal(p["text"]) for p in pages]
        self.head = [normal(p["text"][:80]) for p in pages]
        self.titles = [normal(p["title"]) for p in pages]
        self.index = defaultdict(set)
        for i, text in enumerate(self.norm):
            for word in set(text.split()):
                self.index[word].add(i)

    @classmethod
    def load(cls):
        return cls([json.loads(line) for line in PAGES.open()])

    def matches(self, name):
        """Return {page index: score} for pages containing the exact name."""
        pattern = name_pattern(name)
        if pattern is None:
            return {}
        words = tokens(name)
        anchors = [w for w in words[:-1] if w != "s"]
        pool = set.intersection(*(self.index.get(w, set()) for w in anchors)) if anchors else range(len(self.norm))
        scores = {}
        for i in pool:
            count = len(pattern.findall(self.norm[i]))
            if count:
                scores[i] = count + TITLE_BONUS * bool(pattern.search(self.head[i])) + TITLE_BONUS * bool(pattern.search(self.titles[i]))
        return scores

    def passages(self, name):
        """Top pages for a concept, deterministic, clipped to PAGE_CHARS each."""
        scores = self.matches(name)
        ranked = sorted(scores, key=lambda i: (-scores[i], self.pages[i]["doc"], self.pages[i]["page"]))
        per_doc, picked = Counter(), []
        for i in ranked:
            if per_doc[self.pages[i]["doc"]] < PER_DOC:
                per_doc[self.pages[i]["doc"]] += 1
                picked.append(dict(title=self.pages[i]["title"], text=self.pages[i]["text"][:PAGE_CHARS]))
            if len(picked) == TOP_PAGES:
                break
        return picked


def coverage():
    split = json.loads((SPLIT_DIR / "system_dev.json").read_text())
    library = Library.load()
    counts = {c["name"]: len(library.matches(c["name"])) for c in split["concepts"]}
    n = len(counts)
    print("pages", len(library.pages), "concepts", n, "with any page", sum(v > 0 for v in counts.values()),
          "with 3+ pages", sum(v >= 3 for v in counts.values()), "share", round(sum(v > 0 for v in counts.values()) / n, 3))
    print("no pages:", sorted(k for k, v in counts.items() if not v)[:60])


if __name__ == "__main__":
    {"fetch": fetch, "extract": extract, "coverage": coverage}[sys.argv[1]]()
