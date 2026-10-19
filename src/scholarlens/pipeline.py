
"""End-to-end ScholarLens pipeline + CLI.

Usage:
    python -m scholarlens.pipeline "I am an OBC student ..."
    python -m scholarlens.pipeline "What is the minimum percentage for OBC students?"
"""
from __future__ import annotations

import sys
import time
import csv
import re
from collections import Counter

from . import config, generate
from .profile import QUESTIONS, extract_profile
from .retrieval import Retriever
from .rules import eval_scheme, load_rules
from .utils import load_corpus
from .verify import NLIVerifier, verify_claims


OOS_MSG = (
    "I can only help with the scholarships and schemes in my database "
    "(central and Karnataka). Please describe your situation "
    "(course, category, income, marks) or ask about a specific scheme."
)

DISCLAIMER = (
    "Informational only. Always confirm eligibility and deadlines on the "
    "official portal (National Scholarship Portal / Karnataka State "
    "Scholarship Portal) before applying."
)

SSP_SOURCE_FILE = (
    "Original_Document_5_SSP_Post_Eligibility_Document_2526.pdf"
)


def _is_catalog_question(text: str) -> bool:
    """Detect requests for complete lists of schemes or departments."""
    text = text.lower()

    patterns = [
        (
            r"\b(list|enumerate|show|name)\b.{0,100}"
            r"\b(all|every|schemes?|scholarships?|departments?)\b"
        ),
        (
            r"\b(all|every)\b.{0,60}"
            r"\b(schemes?|scholarships?|departments?)\b"
        ),
        (
            r"\b(schemes?|scholarships?)\b.{0,80}"
            r"\b(departments|names|complete list)\b"
        ),
        (
            r"\b(which|what)\b.{0,60}"
            r"\b(schemes?|scholarships?)\b.{0,60}"
            r"\b(available|included|listed)\b"
        ),
        (
            r"\b(which|what|list|name|show)\b.{0,80}"
            r"\b(departments?|boards?)\b"
        ),
    ]

    return any(re.search(pattern, text) for pattern in patterns)

def _filter_catalog_by_category(rows, question):
    q = question.lower()

    if not re.search(r"\b(scholarships?|schemes?|benefits?)\b", q):
        return rows

    asks_sc = bool(re.search(
        r"\b(sc|scheduled castes?|dalit)\b", q
    ))
    asks_st = bool(re.search(
        r"\b(st|scheduled tribes?|tribal)\b", q
    ))

    # Don't filter questions that don't specify SC or ST.
    if asks_sc == asks_st:
        return rows

    def row_text(row):
        return " ".join(str(v) for v in row.values()).lower()

    filtered = []

    for row in rows:
        text = row_text(row)

        is_sc = bool(re.search(
            r"\bsc\b|scheduled caste|scheduled castes", text
        ))
        is_st = bool(re.search(
            r"\bst\b|scheduled tribe|scheduled tribes", text
        ))

        if asks_sc and is_sc and not is_st:
            filtered.append(row)
        elif asks_st and is_st and not is_sc:
            filtered.append(row)

    return filtered

class ScholarLens:
    def __init__(
        self,
        retrieval_mode: str = "hybrid_rerank",
        verify: str = "nli",
    ):
        """Initialize ScholarLens.

        Args:
            retrieval_mode: Retrieval strategy used by Retriever.
            verify: One of 'nli', 'light', or 'off'.
        """
        if verify not in {"nli", "light", "off"}:
            raise ValueError(
                "verify must be one of: 'nli', 'light', 'off'"
            )

        self.chunks, self.sid2sent = load_corpus()
        self.rules = load_rules()

        if not self.rules:
            raise RuntimeError(
                "No rules found. Run `python -m scholarlens.build_rules` "
                "and verify them."
            )

        self.names = {
            sid: rule["scheme_name"]
            for sid, rule in self.rules.items()
        }

        self.retriever = Retriever(
            self.chunks,
            mode=retrieval_mode,
        )

        self.verify_mode = verify
        self.verifier = NLIVerifier() if verify == "nli" else None

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------

    def _check(self, claims: list[dict]) -> list[dict]:
        """Verify claims against their cited source sentences."""
        if self.verify_mode == "off":
            return [
                {**claim, "supported": None}
                for claim in claims
            ]

        return verify_claims(
            claims,
            self.sid2sent,
            self.verifier,
            config.NLI_THRESHOLD,
        )

    def _sources(self, entries: list[dict]) -> dict:
        """Collect source sentences referenced by scheme entries."""
        sids = set()

        for entry in entries:
            for criterion in entry["criteria"]:
                sids.update(criterion["sids"])

            for claim in entry["claims"]:
                sids.update(claim["sids"])

        return {
            sid: {
                "page": self.sid2sent[sid]["page"],
                "section": self.sid2sent[sid]["section"],
                "text": self.sid2sent[sid]["text"],
            }
            for sid in sids
            if sid in self.sid2sent
        }

    # --------------------------------------------------------------
    # Registry-based catalog questions
    # --------------------------------------------------------------

    def _catalog_answer(
        self,
        user_text: str,
        lat: dict,
        prior_profile: dict | None = None,
    ) -> dict:
        """List Karnataka SSP registry entries without LLM reconstruction.

        Registry entries are not treated as independently verified
        evidence that a scheme is active or its rules are current.
        """
        start = time.time()

        registry_path = (
            config.ROOT
            / "data"
            / "schemes_registry_master.csv"
        )

        if not registry_path.exists():
            lat["catalog"] = round(time.time() - start, 2)
            lat["total"] = round(sum(lat.values()), 2)

            return {
                "intent": "ask_about_scheme",
                "profile": prior_profile or {},
                "answerable": False,
                "claims": [],
                "sources": {},
                "latency": lat,
                "disclaimer": DISCLAIMER,
                "message": (
                    "The master scheme registry was not found. "
                    "Check data/schemes_registry_master.csv."
                ),
            }

        rows = []

        with registry_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                source_file = row.get("file", "").strip()
                scheme_id = row.get("scheme_id", "").strip()

                if (
                    source_file == SSP_SOURCE_FILE
                    and scheme_id.startswith("KA-")
                ):
                    rows.append(row)
            rows = _filter_catalog_by_category(rows, user_text)
        lat["catalog"] = round(time.time() - start, 2)
        lat["total"] = round(sum(lat.values()), 2)

        if not rows:
            return {
                "intent": "ask_about_scheme",
                "profile": prior_profile or {},
                "answerable": False,
                "claims": [],
                "sources": {},
                "latency": lat,
                "disclaimer": DISCLAIMER,
                "message": (
                    "No Karnataka SSP entries matching the configured "
                    "source filename were found in the master registry. "
                    "Check the file column in the CSV."
                ),
            }

        departments = {}

        for row in rows:
            department = (
                row.get("authority", "").strip()
                or "Department not identified"
            )
            departments.setdefault(department, []).append(row)

        claims = []

        for department, entries in departments.items():
            items = []

            for row in entries:
                name = row.get("scheme_name", "").strip()

                if not name:
                    continue

                pages = row.get("source_pages", "").strip()
                verification = (
                    row.get("verification_status", "").strip()
                )

                item = name

                if pages:
                    item += f" (PDF pp. {pages})"

                if verification:
                    item += f" [Verification: {verification}]"

                items.append(item)

            if items:
                claims.append({
                    "claim": (
                        f"**{department}:** "
                        + "; ".join(items)
                    ),
                    "sids": [],
                    "supported": True,
                })

        if not claims:
            return {
                "intent": "ask_about_scheme",
                "profile": prior_profile or {},
                "answerable": False,
                "claims": [],
                "sources": {},
                "latency": lat,
                "disclaimer": DISCLAIMER,
                "message": (
                    "Registry entries were found, but no scheme names "
                    "could be displayed."
                ),
            }

        return {
            "intent": "ask_about_scheme",
            "profile": prior_profile or {},
            "answerable": True,
            "claims": claims,
            "sources": {},
            "latency": lat,
            "disclaimer": DISCLAIMER,
            "message": None,
            "catalog_count": len(rows),
        }

    # --------------------------------------------------------------
    # Main answering pipeline
    # --------------------------------------------------------------

    def answer(
        self,
        user_text: str,
        prior_profile: dict | None = None,
    ) -> dict:
        lat = {}

        # Exhaustive catalog requests use the registry directly.
        if _is_catalog_question(user_text):
            return self._catalog_answer(
                user_text,
                lat,
                prior_profile,
            )

        start = time.time()
        ext = extract_profile(user_text, prior_profile)
        lat["extract"] = round(time.time() - start, 2)

        profile = ext["profile"]

        if ext["intent"] == "out_of_scope":
            lat["total"] = round(sum(lat.values()), 2)
            return {
                "intent": "out_of_scope",
                "profile": profile,
                "message": OOS_MSG,
                "latency": lat,
            }

        if ext["intent"] == "ask_about_scheme":
            return self._qa(ext, lat)

        # 1. Deterministic eligibility evaluation.
        evals = {
            sid: eval_scheme(rule, profile)
            for sid, rule in self.rules.items()
        }

        shown = [
            sid
            for sid, result in evals.items()
            if result["status"] in ("eligible", "needs_info")
        ]

        # 2. Retrieve evidence and generate grounded claims.
        start = time.time()

        gen = (
            generate.generate_claims(
                shown,
                self.names,
                self.retriever,
                self.sid2sent,
            )
            if shown
            else {}
        )

        lat["generate"] = round(time.time() - start, 2)

        # 3. Verify eligibility criteria and generated claims.
        start = time.time()

        entries = []
        dropped = 0
        n_gen = 0
        n_ok = 0

        for sid, result in evals.items():
            criteria = [
                {
                    "text": criterion["text"],
                    "status": criterion["status"],
                    "sids": criterion.get("source_sids", []),
                    "claim": criterion["text"],
                }
                for criterion in result["criteria"]
            ]

            criterion_claims = self._check([
                {
                    "claim": criterion["text"],
                    "sids": criterion["sids"],
                }
                for criterion in criteria
            ])

            for criterion, verified in zip(
                criteria,
                criterion_claims,
            ):
                criterion["verified"] = verified.get("supported")

            claims = self._check(gen.get(sid, []))

            n_gen += len(claims)
            n_ok += sum(
                1
                for claim in claims
                if claim.get("supported")
            )

            if self.verify_mode != "off":
                dropped += sum(
                    1
                    for claim in claims
                    if not claim.get("supported")
                )

                claims = [
                    claim for claim in claims
                    if claim.get("supported")
                ]

            entries.append({
                "scheme_id": sid,
                "scheme_name": self.names[sid],
                "status": result["status"],
                "criteria": criteria,
                "claims": claims,
                "missing_fields": result["missing_fields"],
                "rules_verified": bool(
                    self.rules[sid].get("verified")
                ),
            })

        lat["verify"] = round(time.time() - start, 2)
        lat["total"] = round(sum(lat.values()), 2)

        order = {
            "eligible": 0,
            "needs_info": 1,
            "not_eligible": 2,
        }

        entries.sort(
            key=lambda entry: (
                order[entry["status"]],
                entry["scheme_id"],
            )
        )

        missing = Counter(
            field
            for entry in entries
            if entry["status"] == "needs_info"
            for field in entry["missing_fields"]
        )

        follow_up = [
            QUESTIONS[field]
            for field, _ in missing.most_common(3)
        ]

        return {
            "intent": "find_schemes",
            "profile": profile,
            "schemes": entries,
            "follow_up": follow_up,
            "sources": self._sources(entries),
            "latency": lat,
            "disclaimer": DISCLAIMER,
            "stats": {
                "generated_claims": n_gen,
                "supported_claims": n_ok,
                "dropped_claims": dropped,
                "unverified_rule_files": [
                    sid
                    for sid, rule in self.rules.items()
                    if not rule.get("verified")
                ],
            },
        }

    # --------------------------------------------------------------
    # RAG question answering
    # --------------------------------------------------------------

    def _qa(self, ext: dict, lat: dict) -> dict:
        """Answer a scheme question and diagnose verification outcomes."""
        start = time.time()

        qa = generate.answer_question(
            ext["english_query"],
            self.retriever,
            self.sid2sent,
        )

        raw_claims = qa.get("claims", [])

        # Diagnostics: distinguish generation from verification failure.
        print("\n[DEBUG] Raw QA answerable:", qa.get("answerable"))
        print("[DEBUG] Raw QA claim count:", len(raw_claims))

        checked_claims = self._check(raw_claims)

        print("[DEBUG] Claims after verification:")
        for claim in checked_claims:
            print(
                "  supported =", claim.get("supported"),
                "| entail =", claim.get("entail"),
                "| numbers_ok =", claim.get("numbers_ok"),
                "| reason =", claim.get("reason"),
                "| claim =", claim.get("claim"),
                "| sids =", claim.get("sids"),
            )

        if self.verify_mode != "off":
            claims = [
                claim for claim in checked_claims
                if claim.get("supported") is True
            ]
        else:
            claims = checked_claims

        print("[DEBUG] Final claim count:", len(claims))

        lat["qa"] = round(time.time() - start, 2)
        lat["total"] = round(sum(lat.values()), 2)

        answerable = bool(qa.get("answerable")) and bool(claims)

        sources = {
            sid: {
                "page": self.sid2sent[sid]["page"],
                "section": self.sid2sent[sid]["section"],
                "text": self.sid2sent[sid]["text"],
            }
            for claim in claims
            for sid in claim.get("sids", [])
            if sid in self.sid2sent
        }

        return {
            "intent": "ask_about_scheme",
            "profile": ext["profile"],
            "answerable": answerable,
            "claims": claims,
            "sources": sources,
            "latency": lat,
            "disclaimer": DISCLAIMER,
            "message": (
                None
                if answerable
                else "I could not find this in the scheme documents I have."
            ),
            "stats": {
                "generated_claims": len(raw_claims),
                "verified_claims": sum(
                    1
                    for claim in checked_claims
                    if claim.get("supported") is True
                ),
                "dropped_claims": (
                    len(checked_claims) - len(claims)
                    if self.verify_mode != "off"
                    else 0
                ),
                "verification_mode": self.verify_mode,
            },
        }


# ------------------------------------------------------------------
# Markdown rendering
# ------------------------------------------------------------------

def render_markdown(res: dict) -> str:
    if res["intent"] == "out_of_scope":
        return res["message"]

    lines: list[str] = []
    sources = res.get("sources", {})

    if res["intent"] == "ask_about_scheme":
        if not res.get("answerable", False):
            return res.get(
                "message",
                "I could not find this in the scheme documents I have.",
            )

        if res.get("catalog_count") is not None:
            lines.append(
                f"## Karnataka SSP scheme catalogue "
                f"({res['catalog_count']} registry entries)"
            )
            lines.append(
                "\n*Entries are taken from the master registry. "
                "Page ranges refer to the uploaded SSP PDF. "
                "This list does not establish that every scheme is "
                "currently active or that its rules are up to date.*\n"
            )

        for claim in res.get("claims", []):
            citations = " ".join(
                f"[{sid}]"
                for sid in claim.get("sids", [])
            )
            lines.append(
                f"- {claim['claim']} {citations}".rstrip()
            )

    else:
        icons = {
            "eligible": "✅ Eligible",
            "needs_info": "❓ Need more info",
            "not_eligible": "❌ Not eligible",
        }

        for entry in res["schemes"]:
            status = entry["status"]

            if status == "not_eligible":
                unmet = [
                    criterion
                    for criterion in entry["criteria"]
                    if criterion["status"] == "unmet"
                ]

                lines.append(
                    f"\n**{entry['scheme_name']}** - "
                    f"{icons[status]}"
                )

                lines.extend(
                    f"  - {criterion['text']} "
                    + " ".join(
                        f"[{sid}]"
                        for sid in criterion["sids"]
                    )
                    for criterion in unmet[:2]
                )

                continue

            lines.append(
                f"\n### {entry['scheme_name']} - {icons[status]}"
            )

            for criterion in entry["criteria"]:
                mark = {
                    "met": "✔",
                    "unmet": "✘",
                    "unknown": "?",
                }.get(criterion["status"], "?")

                citations = " ".join(
                    f"[{sid}]"
                    for sid in criterion["sids"]
                )

                lines.append(
                    f"- {mark} {criterion['text']} {citations}".rstrip()
                )

            for claim in entry["claims"]:
                citations = " ".join(
                    f"[{sid}]"
                    for sid in claim["sids"]
                )

                lines.append(
                    f"- {claim['claim']} {citations}".rstrip()
                )

        if res.get("follow_up"):
            lines.append(
                "\n**To narrow this down:** "
                + " ".join(res["follow_up"])
            )

    if sources:
        lines.append("\n**Sources**")

        for sid, source in sorted(
            sources.items(),
            key=lambda item: (
                item[1].get("page", 0),
                item[0],
            ),
        ):
            lines.append(
                f"> [{sid}] (p.{source['page']}) "
                f"{source['text']}"
            )

    if res.get("disclaimer"):
        lines.append(f"\n_{res['disclaimer']}_")

    return "\n".join(lines)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    user_text = (
        " ".join(sys.argv[1:])
        or input("Describe your situation: ")
    )

    result = ScholarLens().answer(user_text)
    print(render_markdown(result))
