"""Text cleaning and normalisation (money, percentages, numbers).

Eligibility text is full of amounts written many ways ("Rs. 2,50,000", "Rs 2.5 lakh",
"INR 250000", "2.5 lakh"). We canonicalise all of them to "Rs 250000" so that regex rules,
retrieval and the numeric-consistency check in verify.py all see one form.
"""
import re
import unicodedata
from typing import Optional

_AMT = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)(?:\s*(lakhs?|lacs?|crores?|L|Cr)\b)?",
    re.I,
)
_BARE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)\b", re.I)
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent\b|percent\b)", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _to_int(num: str, unit: Optional[str]) -> int:
    v = float(num.replace(",", ""))
    u = (unit or "").lower()
    if u.startswith(("lakh", "lac")) or u == "l":
        v *= 1e5
    elif u.startswith("crore") or u == "cr":
        v *= 1e7
    return int(round(v))


def normalize_amounts(text: str) -> str:
    text = _AMT.sub(lambda m: f"Rs {_to_int(m.group(1), m.group(2))}", text)
    text = _BARE.sub(lambda m: f"Rs {_to_int(m.group(1), m.group(2))}", text)
    return text


def normalize_percent(text: str) -> str:
    return _PCT.sub(lambda m: f"{m.group(1)}%", text)


def parse_amount(text: str) -> Optional[int]:
    """First rupee amount in free text, as an int. '2.8 lakh' -> 280000."""
    m = re.search(r"Rs (\d+)", normalize_amounts(text))
    return int(m.group(1)) if m else None


def clean_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    t = re.sub(r"-\n(?=[a-z])", "", t)                       # de-hyphenate line wraps
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"(?im)^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", "", t)   # page-number lines
    t = normalize_amounts(t)
    t = normalize_percent(t)
    return t


def _canon(x: str) -> str:
    f = float(x)
    return str(int(f)) if f == int(f) else str(f)


def extract_numbers(text: str) -> set:
    t = re.sub(r"(?<=\d),(?=\d)", "", normalize_percent(normalize_amounts(text)))
    return {_canon(n) for n in _NUM.findall(t)}
