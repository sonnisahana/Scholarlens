"""Stage 5: deterministic eligibility engine.

A scheme's rules live in data/rules/<scheme_id>.json:
{
  "scheme_id": "...", "scheme_name": "...", "verified": true,
  "criteria": [
    {"id": "c1", "field": "annual_family_income", "op": "<=", "value": 250000,
     "text": "Annual family income must not exceed Rs 250000.",
     "source_sids": ["SCHEME:p2:s7"], "group": null}
  ]
}
Criteria sharing the same non-null "group" are OR-ed (any one satisfied is enough);
everything else is AND-ed. The LLM never decides eligibility: this module does.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from . import config
from .profile import FIELDS

OPS = {"<=", "<", ">=", ">", "==", "in", "not_in", "is_true", "is_false", "contains_any"}


def _n(x):
    return str(x).strip().lower()


def eval_criterion(c: dict, profile: dict) -> str:
    v = profile.get(c["field"])
    if v is None:
        return "unknown"
    op, val = c["op"], c.get("value")
    try:
        if op == "<=":
            ok = v <= val
        elif op == "<":
            ok = v < val
        elif op == ">=":
            ok = v >= val
        elif op == ">":
            ok = v > val
        elif op == "==":
            ok = _n(v) == _n(val)
        elif op == "in":
            ok = _n(v) in {_n(x) for x in val}
        elif op == "not_in":
            ok = _n(v) not in {_n(x) for x in val}
        elif op == "is_true":
            ok = bool(v)
        elif op == "is_false":
            ok = not bool(v)
        elif op == "contains_any":
            ok = any(_n(x) in _n(v) for x in val)
        else:
            return "unknown"
    except TypeError:
        return "unknown"
    return "met" if ok else "unmet"


def eval_scheme(rules: dict, profile: dict) -> dict:
    results = [{**c, "status": eval_criterion(c, profile)} for c in rules["criteria"]]
    groups = defaultdict(list)
    for r in results:
        if r.get("group"):
            groups[r["group"]].append(r["status"])
    statuses = [r["status"] for r in results if not r.get("group")]
    for sts in groups.values():
        statuses.append("met" if "met" in sts else "unknown" if "unknown" in sts else "unmet")
    if "unmet" in statuses:
        status = "not_eligible"
    elif "unknown" in statuses:
        status = "needs_info"
    else:
        status = "eligible"
    missing = sorted({r["field"] for r in results if r["status"] == "unknown"}) if status == "needs_info" else []
    return {"scheme_id": rules["scheme_id"], "status": status, "criteria": results, "missing_fields": missing}


def validate_rules(rules: dict, known_sids: set | None = None) -> list[str]:
    problems = []
    for key in ("scheme_id", "scheme_name", "criteria"):
        if key not in rules:
            problems.append(f"missing key '{key}'")
    for c in rules.get("criteria", []):
        cid = c.get("id", "?")
        if c.get("field") not in FIELDS:
            problems.append(f"{cid}: unknown field {c.get('field')!r}")
        if c.get("op") not in OPS:
            problems.append(f"{cid}: unknown op {c.get('op')!r}")
        if c.get("op") in {"in", "not_in", "contains_any"} and not isinstance(c.get("value"), list):
            problems.append(f"{cid}: op {c.get('op')} needs a list value")
        if not c.get("source_sids"):
            problems.append(f"{cid}: no source_sids (every rule must cite its source sentence)")
        elif known_sids is not None:
            bad = [s for s in c["source_sids"] if s not in known_sids]
            if bad:
                problems.append(f"{cid}: unknown sentence ids {bad}")
    return problems


def load_rules(directory: Path = config.RULES_DIR) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(directory).glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        probs = validate_rules(r)
        if probs:
            raise ValueError(f"{p.name}: " + "; ".join(probs))
        out[r["scheme_id"]] = r
    return out
