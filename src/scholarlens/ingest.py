"""Stages 1-2: PDF -> clean text -> sentences with stable IDs -> section-aware chunks.

Run:  python -m scholarlens.ingest

Every sentence gets an ID like  PMS-SC:p4:s12  (scheme : page : sentence-index-on-page).
That ID is the unit of citation used everywhere downstream.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path

from . import config
from .normalize import clean_text
from .utils import write_jsonl

log = logging.getLogger("ingest")

BULLET = re.compile(r"^\s*(?:[-•●▪*–]|\(?\d+[.)]|\(?[a-z][.)]|\(?[ivx]+\))\s+")
SECTION_KEYWORDS = [
    ("exclusions", r"exclusion|not eligible|ineligib|disqualif|not be entitled"),
    ("eligibility", r"eligib|who can apply|who is entitled|conditions"),
    ("benefits", r"benefit|amount|rate of|financial assistance|stipend|award|scholarship amount"),
    ("documents", r"document|certificate|enclos|proof"),
    ("deadlines", r"deadline|last date|timeline|schedule"),
    ("application", r"how to apply|procedure|application process|apply"),
]


# ----------------------------------------------------------------- PDF extraction

def _tesseract_executable() -> str | None:
    """Find Tesseract even when the IDE terminal has a stale PATH."""
    configured = os.environ.get("TESSERACT_CMD")
    if configured and Path(configured).is_file():
        return configured

    found = shutil.which("tesseract")
    if found:
        return found

    windows_path = Path(r"D:\Engg\7\LLM\tesseract.exe")
    if windows_path.is_file():
        return str(windows_path)

    return None


def _ocr(page, lang: str = "eng") -> str:
    """OCR one PDF page using Tesseract via stdin/stdout."""
    exe = _tesseract_executable()
    if not exe:
        log.warning("Tesseract not found; skipping OCR.")
        return ""

    try:
        pix = page.get_pixmap(dpi=250, alpha=False)
        result = subprocess.run(
            [exe, "stdin", "stdout", "-l", lang, "--psm", "6"],
            input=pix.tobytes("png"),
            capture_output=True,
            timeout=90,
            check=False,
        )

        if result.returncode != 0:
            error = result.stderr.decode("utf-8", errors="replace")
            log.warning("Tesseract failed (%s): %s", lang, error[:300])
            return ""

        return result.stdout.decode("utf-8", errors="replace").strip()

    except Exception as e:
        log.warning("OCR failed (%s): %s", lang, e)
        return ""


def extract_pages(pdf: Path, scheme_id: str | None = None) -> list[str]:
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:
        import fitz

    pages = []
    is_karnataka_ssp = scheme_id == "KA-SSP-SOURCE"

    with fitz.open(pdf) as doc:
        for page_no, page in enumerate(doc, start=1):
            txt = page.get_text("text") or ""

            if is_karnataka_ssp:
                # The SSP PDF contains corrupted embedded Kannada text.
                # OCR every page, not only pages with little extracted text.
                ocr_txt = _ocr(page, lang="kan+eng")
                if ocr_txt:
                    txt = ocr_txt
                else:
                    log.warning(
                        "SSP page %d: OCR failed; using embedded text as fallback.",
                        page_no,
                    )
            elif len(txt.strip()) < 30:
                # Preserve the existing English OCR fallback for other PDFs.
                txt = _ocr(page, lang="eng") or txt

            pages.append(txt)

    return pages



def remove_repeated_lines(pages: list[str]) -> list[str]:
    """Drop running headers/footers: short lines that appear on >= half the pages."""
    counts: Counter = Counter()
    for p in pages:
        for line in {x.strip() for x in p.splitlines() if x.strip()}:
            counts[line] += 1
    thr = max(3, int(0.5 * len(pages)))
    repeated = {l for l, n in counts.items() if n >= thr and len(l) < 120}
    return ["\n".join(l for l in p.splitlines() if l.strip() not in repeated) for p in pages]


# ----------------------------------------------------------------- structure
def looks_like_heading(line: str) -> bool:
    l = line.strip()
    if not l or len(l) > 80 or len(l.split()) > 10 or re.search(r"[.;,]$", l):
        return False
    if re.match(r"^\d+(\.\d+)*[.)]?\s+[A-Z][^.]*$", l) and len(l.split()) <= 7:
        return True
    if BULLET.match(l):
        return False
    return l.isupper() or l.endswith(":") or l.istitle()


def classify_section(heading: str) -> str:
    for name, pat in SECTION_KEYWORDS:
        if re.search(pat, heading, re.I):
            return name
    return "general"


def merge_lines(text: str) -> list[str]:
    """Re-join soft-wrapped lines; keep bullets and headings on their own line."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out.append("")
            continue
        prev = out[-1] if out else ""
        glue = (prev and not BULLET.match(line) and not looks_like_heading(line)
                and not looks_like_heading(prev) and not re.search(r"[.:;?!]$", prev))
        if glue:
            out[-1] = prev + " " + line
        else:
            out.append(line)
    return [l for l in out if l]


_ABBR = re.compile(r"\b(Rs|No|Nos|Dr|Sr|Smt|Shri|Govt|Dept|Min|Max|approx|etc|viz|vs)\.", re.I)


def split_sentences(line: str) -> list[str]:
    s = _ABBR.sub(lambda m: m.group(1) + "\u2024", line)
    for tok in ("i.e.", "e.g."):
        s = s.replace(tok, tok.replace(".", "\u2024"))
    parts = re.split(r"(?<=[.;?!])\s+(?=[A-Z(\d\"'])", s)
    return [p.replace("\u2024", ".").strip() for p in parts if p.strip()]


def build_sentences(scheme_id: str, pages: list[str]) -> list[dict]:
    pages = [clean_text(p) for p in remove_repeated_lines(pages)]
    out, section, stype = [], "Preamble", "general"
    for pno, text in enumerate(pages, 1):
        idx = 0
        for line in merge_lines(text):
            if looks_like_heading(line):
                section = line.strip(" :")
                stype = classify_section(section)
                continue
            for sent in split_sentences(line):
                if len(sent.split()) < 3:
                    continue
                idx += 1
                out.append({"sid": f"{scheme_id}:p{pno}:s{idx}", "scheme_id": scheme_id,
                            "page": pno, "idx": idx, "text": sent,
                            "section": section, "section_type": stype})
    return out


def _windows(group: list[dict], max_sent: int, overlap: int):
    if len(group) <= max_sent:
        yield group
        return
    step = max(1, max_sent - overlap)
    for i in range(0, len(group), step):
        yield group[i:i + max_sent]
        if i + max_sent >= len(group):
            break


def build_chunks(sentences: list[dict], scheme_name: str,
                 max_sent: int = config.CHUNK_MAX_SENT,
                 overlap: int = config.CHUNK_OVERLAP) -> list[dict]:
    chunks, group, n = [], [], 0

    def flush():
        nonlocal n
        for w in _windows(group, max_sent, overlap):
            n += 1
            sid0 = w[0]["scheme_id"]
            chunks.append({
                "chunk_id": f"{sid0}#c{n}", "scheme_id": sid0, "scheme_name": scheme_name,
                "section": w[0]["section"], "section_type": w[0]["section_type"],
                "page": w[0]["page"], "sids": [s["sid"] for s in w],
                "text": f"{scheme_name} - {w[0]['section']}. " + " ".join(s["text"] for s in w),
                "marked": "\n".join(f"[{s['sid']}] {s['text']}" for s in w),
            })

    for s in sentences:
        if group and (s["section"] != group[-1]["section"] or s["scheme_id"] != group[-1]["scheme_id"]):
            flush()
            group = []
        group.append(s)
    if group:
        flush()
    return chunks


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sent", type=int, default=config.CHUNK_MAX_SENT)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(config.REGISTRY, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    all_s, all_c = [], []
    for row in rows:
        sid = row["scheme_id"]
        pdf = config.RAW / (row.get("file") or f"{sid}.pdf")
        if not pdf.exists():
            log.warning("SKIP %-18s (missing %s)", sid, pdf.name)
            continue
        sents = build_sentences(sid, extract_pages(pdf, scheme_id=sid))
        chunks = build_chunks(sents, row["scheme_name"], args.max_sent)
        all_s += sents
        all_c += chunks
        log.info("OK   %-18s %4d sentences %3d chunks", sid, len(sents), len(chunks))
    write_jsonl(config.SENTENCES, all_s)
    write_jsonl(config.CHUNKS, all_c)
    log.info("Wrote %d sentences, %d chunks -> %s", len(all_s), len(all_c), config.PROCESSED)
    log.info("Open data/processed/sentences.jsonl and eyeball it: bad extraction = bad everything downstream.")


if __name__ == "__main__":
    main()
