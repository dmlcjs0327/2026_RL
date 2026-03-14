#!/usr/bin/env python3
"""
Download PDFs for all papers in references.bib into paper/reference_papers/.
Uses known URLs where available, and scrapes PMLR/NeurIPS index pages otherwise.
"""
from __future__ import annotations

import re
import os
import sys
import urllib.request
import ssl
from pathlib import Path

# Optional: use requests if available for better redirect handling
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

BIB_PATH = Path(__file__).resolve().parent / "references.bib"
OUT_DIR = Path(__file__).resolve().parent / "reference_papers"

# Known direct PDF URLs (citation_key -> url). Prefer open-access / official proceedings.
KNOWN_PDF_URLS = {
    "schaul2015uvfa": "http://proceedings.mlr.press/v37/schaul15.pdf",
    "andrychowicz2017her": "https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf",
    "fang2019cher": "https://proceedings.neurips.cc/paper_files/paper/2019/file/83715fd4755b33f9c3958e1a9ee221e1-Paper.pdf",
    "ren2019hgg": "https://proceedings.neurips.cc/paper_files/paper/2019/file/57db7d68d5335b52d5153a4e01adaa6b-Paper.pdf",
    "pong2020skewfit": "http://proceedings.mlr.press/v119/pong20a/pong20a.pdf",
    "ghosh2021gcsl": "https://openreview.net/pdf?id=rALA0Xo6yNJ",
    "eysenbach2021clearning": "https://openreview.net/pdf?id=tc5qisoB-C",
    "liu2022gcrl_survey": "https://ijcai.org/proceedings/2022/0770.pdf",
    "nair2020cig": "http://proceedings.mlr.press/v100/nair20a/nair20a.pdf",
    "mendonca2021lexa": "https://proceedings.neurips.cc/paper_files/paper/2021/file/cc4af25fa9d2d5c953496579b75f6f6c-Paper.pdf",
    "choi2021vgcrl": "http://proceedings.mlr.press/v139/choi21b/choi21b.pdf",
    "eysenbach2022contrastive_gcrl": "https://proceedings.neurips.cc/paper_files/paper/2022/file/e7663e974c4ee7a2b475a4775201ce1f-Paper-Conference.pdf",
    "yuan2024ptgm": "https://proceedings.iclr.cc/paper_files/paper/2024/file/c1842fcfd74f3c2c2a7693994a4a7c37-Paper-Conference.pdf",
    "liu2021rcrl": "https://openreview.net/pdf?id=_TM6rT7tXke",
    "lyle2021aux_rep_dynamics": "http://proceedings.mlr.press/v130/lyle21a/lyle21a.pdf",
    "farebrother2023pvn": "https://openreview.net/pdf?id=oGDKSt9JrZi",
    "yue2023vcr": "https://ojs.aaai.org/index.php/AAAI/article/view/26311/26083",
    "falanga2017narrow_gaps": "https://rpg.ifi.uzh.ch/docs/ICRA17_Falanga.pdf",
    "loquercio2021high_speed_wild": "https://rpg.ifi.uzh.ch/docs/Loquercio21_Science.pdf",
    "song2023cluttered_flight": "https://arxiv.org/pdf/2210.01841.pdf",
    "kaufmann2023swift": "https://rpg.ifi.uzh.ch/docs/Nature23_Kaufmann.pdf",
    "hanover2024drone_racing_survey": "https://rpg.ifi.uzh.ch/docs/arxiv23_hanover.pdf",
    "xing2024bootstrap_rl_agile": "https://rpg.ifi.uzh.ch/docs/CoRL24_Xing.pdf",
    "romero2025acmpc": "https://arxiv.org/pdf/2406.06266.pdf",
}

# Headers to avoid 403
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,*/*",
}


def parse_bib(bib_path: Path) -> list[dict]:
    """Parse references.bib and return list of entries (key, title, year, volume, booktitle, doi)."""
    text = bib_path.read_text(encoding="utf-8", errors="replace")
    entries = []
    current = None
    for line in text.splitlines():
        m = re.match(r"@(?:inproceedings|article)\s*\{\s*([^,]+)\s*,", line)
        if m:
            current = {"key": m.group(1).strip(), "title": "", "year": "", "volume": "", "booktitle": "", "doi": ""}
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
        ym = re.match(r"\s*year\s*=\s*\{\s*(\d+)\s*\},?", line)
        if ym:
            current["year"] = ym.group(1)
            continue
        vm = re.match(r"\s*volume\s*=\s*\{\s*(\d+)\s*\},?", line)
        if vm:
            current["volume"] = vm.group(1)
            continue
        bm = re.match(r"\s*booktitle\s*=\s*\{\s*(.+)\s*\},?", line)
        if bm:
            current["booktitle"] = bm.group(1).strip()
            continue
        jm = re.match(r"\s*journal\s*=\s*\{\s*(.+)\s*\},?", line)
        if jm:
            current["booktitle"] = jm.group(1).strip()
            continue
        dm = re.match(r"\s*doi\s*=\s*\{\s*(.+)\s*\},?", line)
        if dm:
            current["doi"] = dm.group(1).strip()
            continue
    return entries


def normalize_title(t: str) -> str:
    """Normalize title for fuzzy matching."""
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t


def find_pmlr_pdf(volume: str, title: str) -> str | None:
    """Scrape PMLR volume index and return PDF URL for matching title."""
    url = f"https://proceedings.mlr.press/v{volume}/"
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as r:
            html = r.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    target = normalize_title(title)
    # Match pattern: Title text then [[Download PDF](url)] or ](url.pdf)
    # PMLR format: "Title\n...; PMLR vol:pages\n\n[[abs](...)][[Download PDF](http://...pdf)]"
    pdf_re = re.compile(r"\[\[Download PDF\]\]\s*\(\s*(https?://[^)\s]+\.pdf)\s*\)", re.I)
    # Split by paper blocks: look for "Proceedings of" or "PMLR" and then preceding title
    blocks = re.split(r";\s*Proceedings of[^;]*PMLR\s+\d+:\d+-\d+", html)
    for block in blocks:
        if target[:20] not in normalize_title(block):
            continue
        m = pdf_re.search(block)
        if m:
            return m.group(1)
    # Alternative: find all PDF links and titles in order (title often appears before the block)
    titles_found = re.findall(r"([A-Za-z][^;\n]{10,120}?)(?:\s*;\s*|\n)", html)
    links = pdf_re.findall(html)
    # Heuristic: same order as in page
    for i, link in enumerate(links):
        for j in range(max(0, i - 5), min(len(titles_found), i + 5)):
            if j < len(titles_found) and target[:15] in normalize_title(titles_found[j]):
                return link
    return None


def find_neurips_pdf(year: str, title: str) -> str | None:
    """Scrape NeurIPS year index and return PDF URL for matching title."""
    url = f"https://proceedings.neurips.cc/paper_files/paper/{year}"
    if HAS_REQUESTS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            html = r.text
        except Exception:
            return None
    else:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return None
    target = normalize_title(title)
    # Pattern: <a href=".../hash/HASH-Abstract.html">Title</a>
    hash_re = re.compile(r"/hash/([a-f0-9]+)-Abstract\.html")
    link_re = re.compile(r'<a[^>]+href="([^"]*hash[^"]+)"[^>]*>([^<]+)</a>')
    for m in link_re.finditer(html):
        href, link_title = m.group(1), m.group(2)
        if target[:20] in normalize_title(link_title):
            hm = hash_re.search(href)
            if hm:
                h = hm.group(1)
                return f"https://proceedings.neurips.cc/paper_files/paper/{year}/file/{h}-Paper.pdf"
    return None


def download_pdf(url: str, out_path: Path) -> bool:
    """Download PDF from url to out_path. Returns True on success."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if HAS_REQUESTS:
        try:
            r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "")
            if "pdf" not in ct and "octet" not in ct and len(r.content) < 1000:
                return False
            out_path.write_bytes(r.content)
            return True
        except Exception as e:
            print(f"  requests error: {e}")
            return False
    else:
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, context=ssl.create_default_context(), timeout=30) as resp:
                data = resp.read()
                if len(data) < 500:
                    return False
                out_path.write_bytes(data)
                return True
        except Exception as e:
            print(f"  urllib error: {e}")
            return False


def main():
    bib_path = BIB_PATH
    if not bib_path.exists():
        print(f"Bib file not found: {bib_path}")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = parse_bib(bib_path)
    print(f"Found {len(entries)} entries in {bib_path}")
    ok = 0
    failed = []
    for e in entries:
        key = e["key"]
        title = e["title"]
        out_path = OUT_DIR / f"{key}.pdf"
        if out_path.exists():
            print(f"[skip] {key} (already exists)")
            ok += 1
            continue
        url = KNOWN_PDF_URLS.get(key)
        if not url:
            if "PMLR" in (e.get("booktitle") or "") or (e.get("volume") and "Machine Learning" in (e.get("booktitle") or "")):
                url = find_pmlr_pdf(e["volume"], title)
            if not url and "Advances in Neural" in (e.get("booktitle") or ""):
                url = find_neurips_pdf(e["year"], title)
        if url:
            print(f"[downloading] {key} ...")
            if download_pdf(url, out_path):
                print(f"  -> {out_path.name}")
                ok += 1
            else:
                print(f"  -> failed")
                failed.append(key)
        else:
            print(f"[no URL] {key}: {title[:50]}...")
            failed.append(key)
    print(f"\nDone: {ok}/{len(entries)} downloaded. Failed: {len(failed)}")
    if failed:
        print("Missing:", ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
