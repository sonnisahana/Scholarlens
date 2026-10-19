"""Stage 6: grounded, cited explanations (LLM sees ONLY retrieved evidence)."""
from __future__ import annotations

from . import llm
from .retrieval import Retriever

GEN_SYSTEM = """You explain Indian government and scholarship schemes to students in plain, simple English.
You are given EVIDENCE: source sentences, each prefixed by its id in square brackets.
Rules:
1. Use ONLY facts that appear in the EVIDENCE. Never use outside knowledge.
2. Every claim must cite the id(s) of the sentence(s) that support it, copied exactly.
3. Copy numbers (amounts, percentages, dates) exactly as written in the cited sentence. Never compute, sum or round.
4. If the evidence does not cover something, leave it out. Do not guess.
5. At most 4 claims per scheme, each under 30 words. kind is one of: benefit, documents, how_to_apply, deadline, other.
Return ONLY JSON: {"schemes":[{"scheme_id":"...","claims":[{"kind":"benefit","claim":"...","sids":["..."]}]}]}"""




QA_SYSTEM = """You answer questions about scholarship schemes using ONLY
the supplied EVIDENCE sentences.

Rules:
1. Every claim must be supported by its cited source sentence IDs.
2. Keep each claim short and focused on one factual statement.
3. Cite the exact sentence that states the claimed value.
4. Do not cite a sentence merely because it is near the relevant sentence.
5. When different passages give different values, return a separate
   claim for each value, citing only the sentence that states it.
6. Do not combine multiple source passages into one claim.
7. Do not claim that a value applies to a particular scheme unless
   the evidence establishes that connection.
8. Preserve relevant exclusions and exceptions in separate claims.
9. If at least one supplied evidence sentence directly answers
   the question, set answerable to true and return the supported
   claims. Set answerable to false only when the supplied evidence
   does not contain enough information to answer.
10. When the evidence contains different values for the same
    requirement, return a separate claim for each value and
    explicitly flag the discrepancy in your answer.
11. Write claims in clear, natural language. Cite the exact
    sentence containing each value. Do not choose one value as
    authoritative unless the evidence establishes which applies.
12. Use natural wording such as "The minimum required percentage
    is 60%." Do not repeat the word "minimum" and the abbreviation
    "min" in the same claim.

Return ONLY valid JSON:
{"answerable": true, "claims": [{"claim": "...", "sids": ["..."]}]}"""



def _ordered(sids, sid2sent):
    return sorted(set(sids), key=lambda s: (sid2sent[s]["page"], sid2sent[s]["idx"]))


def _clean_claims(items, allowed: set) -> list[dict]:
    out = []
    for c in items or []:
        text = str(c.get("claim", "")).strip()
        if text:
            out.append({"kind": c.get("kind", "other"), "claim": text,
                        "sids": [s for s in c.get("sids", []) if s in allowed],
                        "raw_sids": c.get("sids", [])})
    return out


def generate_claims(scheme_ids: list[str], names: dict, retriever: Retriever, sid2sent: dict,
                    batch: int = 6) -> dict[str, list[dict]]:
    """Return {scheme_id: [claim dicts]} for the schemes we want to explain."""
    result = {sid: [] for sid in scheme_ids}
    for i in range(0, len(scheme_ids), batch):
        group, allowed, blocks = scheme_ids[i:i + batch], {}, []
        for sid in group:
            chunks = retriever.search(f"{names[sid]} scholarship amount benefits documents required "
                                      f"how to apply last date", k=3, scheme_id=sid)
            sids = _ordered([s for ch in chunks for s in ch["sids"]], sid2sent)
            allowed[sid] = set(sids)
            blocks.append(f"### SCHEME {sid} - {names[sid]}\n" +
                          "\n".join(f"[{s}] {sid2sent[s]['text']}" for s in sids))
        out = llm.complete_json(GEN_SYSTEM, "EVIDENCE:\n\n" + "\n\n".join(blocks), max_tokens=2500)
        for item in out.get("schemes", []):
            sid = item.get("scheme_id")
            if sid in allowed:
                result[sid] += _clean_claims(item.get("claims"), allowed[sid])
    return result



def answer_question(
    question: str,
    retriever: Retriever,
    sid2sent: dict,
    k: int = 6,
) -> dict:
    """
    Retrieve evidence using multiple query formulations, then generate
    an answer grounded exclusively in the combined evidence.

    Category/threshold questions receive additional retrieval coverage.
    """
    import re

    queries = [question]
    q = question.lower()

    # Expand searches for category-specific marks/percentage questions.
    asks_about_marks = bool(
        re.search(
            r"\b(minimum|minimum percentage|percentage|marks|"
            r"cut.?off|pass mark|qualif(?:y|ying|ication))\b",
            q,
        )
    )

    asks_about_obc = bool(
        re.search(r"\b(obc|backward class|category.?i|category 1)\b", q)
    )

    if asks_about_marks and asks_about_obc:
        queries.extend([
            "OBC minimum marks percentage eligibility criteria",
            "Category I minimum marks percentage eligibility",
            "other OBC categories minimum marks percentage eligibility",
            "minimum percentage marks category-wise eligibility exclusions",
        ])
    elif asks_about_marks:
        queries.extend([
            f"{question} eligibility criteria minimum marks",
            f"{question} category-wise percentage requirements",
            "minimum qualifying percentage marks eligibility exceptions",
        ])

    # Retrieve candidates for each query. BM25 provides a lexical
    # retrieval path alongside the configured hybrid/dense retrieval.
    retrieved = {}
    for query in queries:
        for mode in (None, "bm25"):
            try:
                chunks = retriever.search(
                    query,
                    k=max(k, 6),
                    mode=mode,
                )
            except Exception:
                # If one retrieval mode fails, keep results from the other.
                continue

            for rank, chunk in enumerate(chunks):
                chunk_id = chunk.get("chunk_id")
                key = chunk_id or chunk.get("text", "")

                if not key:
                    continue

                if key not in retrieved:
                    retrieved[key] = {
                        "chunk": chunk,
                        "hits": 0,
                        "best_rank": rank,
                    }

                retrieved[key]["hits"] += 1
                retrieved[key]["best_rank"] = min(
                    retrieved[key]["best_rank"],
                    rank,
                )

    # Prefer chunks retrieved by multiple queries, breaking ties by
    # the best rank achieved. Keep the evidence set bounded.
    ranked = sorted(
        retrieved.values(),
        key=lambda item: (
            -item["hits"],
            item["best_rank"],
        ),
    )

    max_chunks = max(12, k * 2)
    selected = [item["chunk"] for item in ranked[:max_chunks]]

    sids = _ordered(
        [
            sid
            for chunk in selected
            for sid in chunk.get("sids", [])
            if sid in sid2sent
        ],
        sid2sent,
    )

    # Focus OBC percentage questions on category-specific percentage evidence.
    if asks_about_marks and asks_about_obc:
        targeted_sids = []

        for sid in sids:
            text = sid2sent[sid]["text"].lower()

            has_percentage = bool(
                re.search(r"\d+\s*%|\bmin\b|minimum|ಕನಿಷ್ಠ", text)
            )

            has_category = any(
                term in text
                for term in (
                    "category-1",
                    "category 1",
                    "category-i",
                    "cat-i",
                    "other obc",
                    "obc",
                    "ಹಿಂದುಳಿದ ವರ್ಗಗಳು",
                    "ಪ್ರವರ್ಗ-1",
                )
            )

            if has_percentage and has_category:
                targeted_sids.append(sid)

        # Keep the original evidence if the targeted filter finds nothing.
        if targeted_sids:
            sids = _ordered(targeted_sids, sid2sent)

    allowed = set(sids)

    if not sids:
        return {"answerable": False, "claims": []}

    evidence = "\n".join(
        f"[{sid}] {sid2sent[sid]['text']}"
        for sid in sids
    )

    out = llm.complete_json(
        QA_SYSTEM,
        f"QUESTION: {question}\n\nEVIDENCE:\n{evidence}",
        max_tokens=1200,
    )

    print("\n[DEBUG] Evidence sentence count:", len(sids))
    print("[DEBUG] Evidence for QA:")
    print(evidence)
    print("[DEBUG] Raw LLM JSON response:")
    print(out)

    claims = _clean_claims(out.get("claims"), allowed)

    # Keep only claims with at least one valid retrieved citation.
    claims = [
        claim for claim in claims
        if claim["sids"]
    ]

    return {
        "answerable": bool(out.get("answerable")) and bool(claims),
        "claims": claims,
    }
