"""Stage 9: evaluation + ablations.   python -m scholarlens.evaluate --task all [--limit 20] [--oracle-profile]

tasks
  retrieval     Recall@k / MRR / nDCG@k for tfidf, bm25, dense, hybrid, hybrid_rerank
  extraction    field-level accuracy of profile extraction
  eligibility   P/R/F1 + false-eligible rate: A) no-RAG  B) RAG-only LLM  C) RAG + rules
  groundedness  claim support before vs after the verifier (judged by an independent LLM judge),
                citation recall, refusal / follow-up behaviour, latency

Inputs:  data/gold_set/personas.jsonl, data/gold_set/retrieval_queries.jsonl   (rows with "TEMPLATE": true are skipped)
Outputs: results/*.json and results/summary.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import time

from . import config, variants
from .pipeline import ScholarLens
from .profile import FIELDS, extract_profile
from .retrieval import Retriever
from .rules import eval_scheme
from .utils import load_corpus, read_jsonl
from .verify import llm_judge


def _gold(path, limit):
    rows = [r for r in read_jsonl(path) if not r.get("TEMPLATE")]
    if not rows:
        raise SystemExit(f"No usable rows in {path} (template rows are skipped). Write your gold set first.")
    return rows[:limit] if limit else rows


def _save(name, obj):
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------ retrieval
def eval_retrieval(limit):
    chunks, _ = load_corpus()
    rows = _gold(config.GOLD_RETRIEVAL, limit)
    ks, table, detail = (1, 3, 5), [], {}
    for mode in Retriever.MODES:
        r = Retriever(chunks, mode=mode)
        agg = {f"R@{k}": [] for k in ks} | {"MRR": []} | {f"nDCG@{k}": [] for k in ks}
        for row in rows:
            rel = set(row["relevant_sids"])
            res = r.search(row["query"], k=max(ks), scheme_id=row.get("scheme_id"))
            hit = [1 if rel & set(c["sids"]) else 0 for c in res]
            n_rel = sum(1 for c in chunks if (row.get("scheme_id") in (None, c["scheme_id"])) and rel & set(c["sids"]))
            for k in ks:
                cov = rel & {s for c in res[:k] for s in c["sids"]}
                agg[f"R@{k}"].append(len(cov) / len(rel))
                dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hit[:k]))
                idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k))) or 1
                agg[f"nDCG@{k}"].append(dcg / idcg)
            agg["MRR"].append(next((1 / (i + 1) for i, h in enumerate(hit) if h), 0.0))
        means = {m: round(sum(v) / len(v), 3) for m, v in agg.items()}
        detail[mode] = means
        table.append([mode] + [means[m] for m in agg])
    _save("retrieval", detail)
    return "### Retrieval ablation\n" + _table(["mode"] + list(agg), table)


# ------------------------------------------------------------------ extraction
def eval_extraction(limit):
    rows = [r for r in _gold(config.GOLD_PERSONAS, limit) if r.get("profile")]
    ok = {f: [0, 0] for f in FIELDS}
    halluc = n_null = 0
    for row in rows:
        pred = extract_profile(row["text"])["profile"]
        for f in FIELDS:
            g, p = row["profile"].get(f), pred.get(f)
            if g is None:
                n_null += 1
                halluc += p is not None
            else:
                ok[f][1] += 1
                same = (abs(float(g) - float(p)) <= 0.5) if isinstance(g, (int, float)) and not isinstance(g, bool) \
                    and p is not None else str(g).lower() == str(p).lower()
                ok[f][0] += bool(same)
    per = {f: round(a / b, 3) for f, (a, b) in ok.items() if b}
    tot_a, tot_b = sum(a for a, _ in ok.values()), sum(b for _, b in ok.values())
    res = {"per_field_accuracy": per, "micro_accuracy": round(tot_a / max(tot_b, 1), 3),
           "hallucinated_field_rate": round(halluc / max(n_null, 1), 3)}
    _save("extraction", res)
    return ("### Profile extraction\n" + _table(["field", "accuracy"], per.items()) +
            f"\n\nMicro accuracy **{res['micro_accuracy']}**, hallucinated-field rate **{res['hallucinated_field_rate']}**")


# ------------------------------------------------------------------ eligibility
def _prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "false_eligible_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0}


def eval_eligibility(limit, oracle):
    lens = ScholarLens(retrieval_mode="hybrid_rerank", verify="off")
    rows = [r for r in _gold(config.GOLD_PERSONAS, limit) if "gold_eligible" in r]
    ids = list(lens.names)
    systems = {"A: no RAG": [0, 0, 0, 0], "B: RAG only (LLM decides)": [0, 0, 0, 0], "C: RAG + rules": [0, 0, 0, 0]}
    preds = []
    for row in rows:
        profile = row["profile"] if (oracle and row.get("profile")) else extract_profile(row["text"])["profile"]
        out = {"A: no RAG": variants.no_rag(row["text"], lens.names),
               "B: RAG only (LLM decides)": variants.rag_only(row["text"], lens.names, lens.retriever, lens.sid2sent),
               "C: RAG + rules": {sid: eval_scheme(lens.rules[sid], profile)["status"] for sid in ids}}
        gold = set(row["gold_eligible"])
        for name, status in out.items():
            c = systems[name]
            for sid in ids:
                pred, g = status[sid] == "eligible", sid in gold
                c[0] += pred and g
                c[1] += pred and not g
                c[2] += (not pred) and g
                c[3] += (not pred) and (not g)
        preds.append({"id": row["id"], "gold": sorted(gold),
                      **{n: sorted(s for s, v in o.items() if v == "eligible") for n, o in out.items()}})
    res = {n: _prf(*c) for n, c in systems.items()}
    _save("eligibility", {"systems": res, "oracle_profile": oracle, "per_persona": preds})
    return (f"### Eligibility decisions ({'oracle profile' if oracle else 'end-to-end'})\n" +
            _table(["system", "precision", "recall", "F1", "false-eligible rate"],
                   [[n, *v.values()] for n, v in res.items()]))


# ------------------------------------------------------------------ groundedness
def eval_groundedness(limit):
    rows = _gold(config.GOLD_PERSONAS, limit)
    off, on = ScholarLens(verify="off"), ScholarLens(verify="nli")
    sid2sent = on.sid2sent
    before = [0, 0]
    after = [0, 0]
    cited = total = 0
    oos_ok = oos_n = fu_ok = fu_n = 0
    lat = []
    for row in rows:
        t0 = time.time()
        r_on = on.answer(row["text"])
        lat.append(time.time() - t0)
        if row.get("expected_intent") == "out_of_scope":
            oos_n += 1
            oos_ok += r_on["intent"] == "out_of_scope"
        if "missing_info" in row.get("tags", []):
            fu_n += 1
            fu_ok += bool(r_on.get("follow_up"))
        if r_on["intent"] != "find_schemes":
            continue
        r_off = off.answer(row["text"])
        gen_off = [c for e in r_off["schemes"] for c in e["claims"]]
        gen_on = [c for e in r_on["schemes"] for c in e["claims"]]
        for claims, acc in ((gen_off, before), (gen_on, after)):
            if claims:
                v = llm_judge(claims, sid2sent)
                acc[0] += sum(v)
                acc[1] += len(v)
        cited += sum(1 for c in gen_off if c["sids"])
        total += len(gen_off)
    res = {"groundedness_before_verifier": round(before[0] / max(before[1], 1), 3),
           "groundedness_after_verifier": round(after[0] / max(after[1], 1), 3),
           "claims_before": before[1], "claims_after": after[1],
           "citation_recall": round(cited / max(total, 1), 3),
           "out_of_scope_refusal_accuracy": round(oos_ok / oos_n, 3) if oos_n else None,
           "follow_up_rate_on_missing_info": round(fu_ok / fu_n, 3) if fu_n else None,
           "latency_mean_s": round(st.mean(lat), 2),
           "latency_p95_s": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2)}
    _save("groundedness", res)
    return "### Groundedness and behaviour\n" + _table(["metric", "value"], res.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all", choices=["all", "retrieval", "extraction", "eligibility", "groundedness"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--oracle-profile", action="store_true", help="use gold profiles (isolates rule/RAG quality)")
    a = ap.parse_args()
    parts = []
    if a.task in ("all", "retrieval"):
        parts.append(eval_retrieval(a.limit))
    if a.task in ("all", "extraction"):
        parts.append(eval_extraction(a.limit))
    if a.task in ("all", "eligibility"):
        parts.append(eval_eligibility(a.limit, a.oracle_profile))
    if a.task in ("all", "groundedness"):
        parts.append(eval_groundedness(a.limit))
    md = "\n\n".join(parts)
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
