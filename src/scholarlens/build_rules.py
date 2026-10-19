"""Draft eligibility rules with an LLM, then YOU verify them by hand.

Run:  python -m scholarlens.build_rules            (skips schemes that already have a rules file)
      python -m scholarlens.build_rules --force

Output: data/rules/<scheme_id>.json with "verified": false.
Open each file next to the PDF, fix mistakes, then set "verified": true.
The hand-verified rules are also your ground truth for the evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re

from . import config, llm
from .profile import ENUMS, FIELDS
from .rules import OPS, validate_rules
from .utils import read_jsonl

ELIG_KW = re.compile(r"eligib|income|marks|percent|%|domicile|resident|age|category|caste|class|course|"
                     r"degree|disab|gender|girl|woman|women|minority|not be|only|must|should", re.I)

SYSTEM = f"""You convert scholarship eligibility text into machine-checkable criteria.
Input: numbered source sentences, each prefixed with its id in [brackets].
Return ONLY JSON: {{"criteria": [...], "unmapped_clauses": [...]}}
Each criterion: {{"id":"c1","field":<one of the fields>,"op":<one of the ops>,"value":<value>,
"text":"<the requirement in one plain sentence, copying numbers exactly>",
"source_sids":["<ids of the sentences that state it>"],"group":null}}
FIELDS (and allowed values):
{json.dumps(FIELDS, indent=1)}
ENUM VALUES: {json.dumps(ENUMS)}
OPS: {sorted(OPS)}   ("in"/"not_in"/"contains_any" take a list; "is_true"/"is_false" take no value)
Rules:
- Only create a criterion if it is explicitly stated in the sentences. Never use outside knowledge.
- Income limits: field annual_family_income, op "<=", value in integer rupees.
- If several alternatives are each sufficient (e.g. income below X OR belongs to category Y), give those
  criteria the same non-null "group" string (e.g. "g1"). Otherwise group is null.
- A requirement that cannot be expressed with the fields (e.g. "must not receive another scholarship")
  goes into "unmapped_clauses" as a string with its sentence id.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    sents = read_jsonl(config.SENTENCES)
    if not sents:
        raise SystemExit("Run `python -m scholarlens.ingest` first.")
    known = {s["sid"] for s in sents}
    with open(config.REGISTRY, encoding="utf-8") as f:
        registry = {r["scheme_id"]: r for r in csv.DictReader(f)}
    config.RULES_DIR.mkdir(parents=True, exist_ok=True)

    for sid in sorted({s["scheme_id"] for s in sents}):
        path = config.RULES_DIR / f"{sid}.json"
        if path.exists() and not args.force:
            print(f"keep  {sid} (exists)")
            continue
        mine = [s for s in sents if s["scheme_id"] == sid]
        rel = [s for s in mine if s["section_type"] in ("eligibility", "exclusions") or ELIG_KW.search(s["text"])][:150]
        user = "SENTENCES:\n" + "\n".join(f"[{s['sid']}] {s['text']}" for s in rel)
        out = llm.complete_json(SYSTEM, user, max_tokens=3000)
        rules = {"scheme_id": sid, "scheme_name": registry[sid]["scheme_name"], "verified": False,
                 "criteria": out.get("criteria", []), "unmapped_clauses": out.get("unmapped_clauses", [])}
        for i, c in enumerate(rules["criteria"], 1):
            c.setdefault("id", f"c{i}")
            c.setdefault("group", None)
        problems = validate_rules(rules, known)
        if problems:                      # drop invalid criteria but tell the human
            print(f"WARN  {sid}: " + "; ".join(problems))
            bad = {p.split(":")[0] for p in problems}
            rules["criteria"] = [c for c in rules["criteria"] if c["id"] not in bad]
        path.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {sid}: {len(rules['criteria'])} criteria  -> VERIFY BY HAND, then set verified=true")


if __name__ == "__main__":
    main()
