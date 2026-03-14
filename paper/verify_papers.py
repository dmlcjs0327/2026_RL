#!/usr/bin/env python3
"""
Verify that each PDF in reference_papers/ matches the corresponding bib entry
(title and first author) by extracting text from the first 2 pages.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

BIB_PATH = Path(__file__).resolve().parent / "references.bib"
REF_DIR = Path(__file__).resolve().parent / "reference_papers"

# Stop words and very short tokens to exclude from title-word matching
STOP = {"the", "and", "for", "with", "via", "from", "using", "based", "learning", "reinforcement"}


def parse_bib(bib_path: Path) -> list[dict]:
    """Parse references.bib; return list of {key, title, year, author_raw, first_author_last}."""
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries = []
    current = None
    for line in text.splitlines():
        m = re.match(r"@(?:inproceedings|article)\s*\{\s*([^,]+)\s*,", line)
        if m:
            current = {
                "key": m.group(1).strip(),
                "title": "",
                "year": "",
                "author_raw": "",
                "first_author_last": "",
            }
            continue
        if current is None:
            continue
        if line.strip().rstrip('"') == "}":
            if current.get("title"):
                entries.append(current)
            current = None
            continue
        tm = re.match(r"\s*title\s*=\s*\{\s*(.+)\s*\},?", line)
        if tm:
            current["title"] = re.sub(r"[\{\}]", "", tm.group(1).strip()).replace("--", "-")
            continue
        am = re.match(r"\s*author\s*=\s*\{\s*(.+)\s*\},?", line)
        if am:
            raw = re.sub(r"[\{\}]", "", am.group(1).strip())
            current["author_raw"] = raw
            # First author: "Last, First" or "Last, First M."
            first = raw.split(" and ")[0].strip()
            if "," in first:
                current["first_author_last"] = first.split(",")[0].strip().lower()
            else:
                current["first_author_last"] = first.split()[-1].strip().lower() if first else ""
            continue
        ym = re.match(r"\s*year\s*=\s*\{\s*(\d+)\s*\},?", line)
        if ym:
            current["year"] = ym.group(1)
            continue
    return entries


def normalize_for_match(s: str) -> str:
    """Lowercase, collapse spaces, remove non-alphanumeric for matching."""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    # Fix PDFs that insert spaces inside words (e.g. "R ECURSIVE" -> "recursive")
    s = re.sub(r"\b([a-z]) ([a-z])", r"\1\2", s)
    return " ".join(s.split())


def title_significant_words(title: str) -> set[str]:
    """Set of significant (length >= 2, not stop) words from title."""
    norm = normalize_for_match(title)
    words = {w for w in norm.split() if len(w) >= 2 and w not in STOP}
    return words


def extract_pdf_text(pdf_path: Path, max_pages: int = 2) -> str:
    """Extract text from first max_pages of PDF using pdftotext."""
    try:
        out = subprocess.run(
            ["pdftotext", "-l", str(max_pages), str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if out.returncode != 0:
            return ""
        return (out.stdout or "") + (out.stderr or "")
    except Exception:
        return ""


def verify_one(entry: dict, pdf_path: Path) -> tuple[str, str]:
    """
    Verify PDF matches bib entry. Returns (status, detail).
    status: "OK", "WARN", "FAIL"
    """
    if not pdf_path.exists():
        return "FAIL", "PDF file not found"
    raw_text = extract_pdf_text(pdf_path)
    if not raw_text or len(raw_text.strip()) < 100:
        return "FAIL", "Could not extract text (corrupt or empty PDF?)"
    text_norm = normalize_for_match(raw_text)
    title_words = title_significant_words(entry["title"])
    if not title_words:
        return "WARN", "No significant title words to check"
    found = sum(1 for w in title_words if w in text_norm)
    ratio = found / len(title_words)
    first_author = (entry.get("first_author_last") or "").strip()
    author_ok = first_author in text_norm if first_author else True

    if ratio >= 0.7 and author_ok:
        return "OK", f"title {found}/{len(title_words)} words, author present"
    if ratio >= 0.5 and author_ok:
        return "WARN", f"title {found}/{len(title_words)} words (partial match), author present"
    if ratio >= 0.7 and not author_ok:
        return "WARN", f"title {found}/{len(title_words)} words; first author '{first_author}' not found in first 2 pages"
    if ratio < 0.5:
        missing = title_words - {w for w in title_words if w in text_norm}
        return "FAIL", f"title only {found}/{len(title_words)} words in PDF; missing e.g. {list(missing)[:5]}"
    return "WARN", f"title {found}/{len(title_words)}, author_ok={author_ok}"


def main():
    if not BIB_PATH.exists():
        print(f"Bib not found: {BIB_PATH}")
        return 1
    if not REF_DIR.exists():
        print(f"Directory not found: {REF_DIR}")
        return 1
    entries = parse_bib(BIB_PATH)
    key2entry = {e["key"]: e for e in entries}
    pdfs = sorted(REF_DIR.glob("*.pdf"))
    results = []
    for pdf_path in pdfs:
        key = pdf_path.stem
        entry = key2entry.get(key)
        if not entry:
            results.append((key, "FAIL", f"Key '{key}' not in references.bib", entry))
            continue
        status, detail = verify_one(entry, pdf_path)
        results.append((key, status, detail, entry))
    # Also check for bib keys without PDF
    keys_with_pdf = {p.stem for p in pdfs}
    for key in key2entry:
        if key not in keys_with_pdf:
            results.append((key, "FAIL", "No PDF file for this bib entry", key2entry[key]))

    # Report
    print("Bib–PDF verification report\n" + "=" * 60)
    ok = sum(1 for _, s, _, _ in results if s == "OK")
    warn = sum(1 for _, s, _, _ in results if s == "WARN")
    fail = sum(1 for _, s, _, _ in results if s == "FAIL")
    for key, status, detail, entry in sorted(results, key=lambda x: (x[1] != "OK", x[0])):
        title_short = (entry["title"][:50] + "…") if entry and len(entry["title"]) > 50 else (entry["title"] if entry else key)
        symbol = {"OK": "✓", "WARN": "?", "FAIL": "✗"}[status]
        print(f"  [{symbol}] {key}")
        print(f"      Bib: {title_short}")
        print(f"      {status}: {detail}")
        print()
    print("=" * 60)
    print(f"Summary: OK={ok}, WARN={warn}, FAIL={fail}  (total entries: {len(entries)}, PDFs: {len(pdfs)})")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    exit(main())
