"""Baseline systems for the headline ablation (A: no RAG, B: RAG-only LLM judgement)."""
from __future__ import annotations

from . import llm
from .retrieval import Retriever

LABELS = ("eligible", "not_eligible", "needs_info")
SYS_A = ("You are an expert on Indian central and Karnataka scholarship schemes. Using your own knowledge, decide "
         "for each listed scheme whether the student is eligible. If key information is missing say needs_info. "
         'Return ONLY JSON: {"<scheme_id>": "eligible|not_eligible|needs_info", ...}')
SYS_B = ("Decide, for each scheme, whether the student is eligible using ONLY the provided scheme text. "
         "If a needed fact about the student is missing say needs_info. "
         'Return ONLY JSON: {"<scheme_id>": "eligible|not_eligible|needs_info", ...}')


def _norm(out: dict, ids) -> dict:
    return {i: (out.get(i) if out.get(i) in LABELS else "not_eligible") for i in ids}


def no_rag(student_text: str, names: dict) -> dict:
    listing = "\n".join(f"- {sid}: {n}" for sid, n in names.items())
    out = llm.complete_json(SYS_A, f"STUDENT: {student_text}\n\nSCHEMES:\n{listing}", max_tokens=1200)
    return _norm(out, names)


def rag_only(student_text: str, names: dict, retriever: Retriever, sid2sent: dict, k: int = 4) -> dict:
    blocks = []
    for sid, n in names.items():
        chunks = retriever.search(f"eligibility conditions income category marks course {student_text}",
                                  k=k, scheme_id=sid)
        blocks.append(f"### {sid} - {n}\n" + "\n".join(c["marked"] for c in chunks))
    out = llm.complete_json(SYS_B, f"STUDENT: {student_text}\n\nSCHEME TEXT:\n" + "\n\n".join(blocks), max_tokens=1200)
    return _norm(out, names)
