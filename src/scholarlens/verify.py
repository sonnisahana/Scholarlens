"""Stage 7: claim verification (citation validity + number check + NLI entailment).

Two independent checkers are provided on purpose:
  * NLIVerifier  - cheap, runs inside the pipeline to FILTER unsupported claims.
  * llm_judge    - used in evaluation to MEASURE groundedness (so we do not grade the filter with itself).
"""
from __future__ import annotations

import numpy as np

from . import config, llm
from .normalize import extract_numbers


def numbers_supported(claim: str, premise: str) -> bool:
    """Every number in the claim must appear in the cited text."""
    return extract_numbers(claim) <= extract_numbers(premise)


class NLIVerifier:
    def __init__(self, model_name: str = config.NLI_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)
        id2label = self.model.model.config.id2label
        self.ent_idx = next(int(i) for i, l in id2label.items() if str(l).lower().startswith("entail"))

    def entail(self, premise: str, hypothesis: str) -> float:
        logits = np.asarray(self.model.predict([(premise, hypothesis)]))[0]
        e = np.exp(logits - logits.max())
        return float((e / e.sum())[self.ent_idx])


def verify_claims(claims: list[dict], sid2sent: dict, verifier: NLIVerifier | None,
                  threshold: float = config.NLI_THRESHOLD) -> list[dict]:
    """Annotate each claim with supported / entail / numbers_ok / reason (in place)."""
    for c in claims:
        sids = [s for s in c.get("sids", []) if s in sid2sent]
        if not sids:
            c.update(supported=False, entail=0.0, numbers_ok=False, reason="no_valid_citation")
            continue
        premise = " ".join(sid2sent[s]["text"] for s in sids)
        nums = numbers_supported(c["claim"], premise)
        p = verifier.entail(premise, c["claim"]) if verifier else None
        ok_nli = True if p is None else p >= threshold
        reason = "ok" if (nums and ok_nli) else ("number_mismatch" if not nums else "low_entailment")
        c.update(supported=bool(nums and ok_nli), entail=p, numbers_ok=nums, reason=reason)
    return claims


JUDGE_SYS = ("You audit claims against source text. For each item decide whether the CLAIM is fully supported "
             "by the SOURCE (every fact and every number). Be strict. "
             'Return ONLY JSON: {"results":[{"i":0,"supported":true}]}')


def llm_judge(claims: list[dict], sid2sent: dict, batch: int = 25) -> list[bool]:
    verdicts: list[bool] = []
    for start in range(0, len(claims), batch):
        part, items = claims[start:start + batch], []
        for i, c in enumerate(part):
            prem = " ".join(sid2sent[s]["text"] for s in c.get("sids", []) if s in sid2sent)
            items.append(f"#{i}\nSOURCE: {prem or '(none)'}\nCLAIM: {c['claim']}")
        out = llm.complete_json(JUDGE_SYS, "\n\n".join(items), max_tokens=1500)
        m = {int(r["i"]): bool(r["supported"]) for r in out.get("results", []) if "i" in r}
        verdicts += [m.get(i, False) for i in range(len(part))]
    return verdicts
