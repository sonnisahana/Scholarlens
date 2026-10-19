"""Offline tests: no PDFs, no models, no API key needed.   pytest -q"""
from scholarlens.ingest import build_chunks, build_sentences, split_sentences
from scholarlens.normalize import clean_text, extract_numbers, normalize_amounts, parse_amount
from scholarlens.profile import clean_profile, merge_profiles
from scholarlens.retrieval import Retriever, rrf
from scholarlens.rules import eval_scheme, validate_rules
from scholarlens.verify import numbers_supported


def test_amount_variants_all_canonicalise():
    for s in ["Rs. 2,50,000", "Rs 2.5 lakh", "INR 250000", "₹2.5 L", "2.5 lakh", "Rs.2.5 Lakhs"]:
        assert normalize_amounts(s) == "Rs 250000", s


def test_parse_amount_from_chat_text():
    assert parse_amount("my family income is 2.8 lakh per year") == 280000
    assert parse_amount("no money info here") is None


def test_clean_text_percent_and_pageno():
    out = clean_text("Minimum 60 per cent marks.\nPage 3 of 10\nIncome Rs. 1,00,000 only.")
    assert "60%" in out and "Page 3" not in out and "Rs 100000" in out


def test_sentence_split_keeps_abbreviations():
    s = split_sentences("Income must be below Rs 100000 per annum. The student should be i.e. a resident. Apply online.")
    assert len(s) == 3


def test_sentence_ids_and_chunks():
    page = ("ELIGIBILITY CONDITIONS\n"
            "1. The applicant must be a resident of Karnataka.\n"
            "2. Annual family income must not exceed Rs. 2.5 lakh.\n"
            "3. The applicant must have secured at least 50 per cent marks.\n\n"
            "BENEFITS\nThe scholarship amount is Rs. 20,000 per year.")
    sents = build_sentences("T1", [page])
    assert sents[0]["sid"] == "T1:p1:s1" and sents[0]["section_type"] == "eligibility"
    assert any("Rs 250000" in s["text"] for s in sents)
    assert any(s["section_type"] == "benefits" for s in sents)
    chunks = build_chunks(sents, "Test Scheme", max_sent=2, overlap=1)
    assert all(c["sids"] and c["marked"].startswith("[T1:") for c in chunks)
    covered = {s for c in chunks for s in c["sids"]}
    assert covered == {s["sid"] for s in sents}


RULES = {"scheme_id": "X", "scheme_name": "X", "criteria": [
    {"id": "c1", "field": "annual_family_income", "op": "<=", "value": 250000, "source_sids": ["X:p1:s1"]},
    {"id": "c2", "field": "category", "op": "in", "value": ["SC", "ST"], "source_sids": ["X:p1:s2"], "group": "g"},
    {"id": "c3", "field": "disability_percent", "op": ">=", "value": 40, "source_sids": ["X:p1:s3"], "group": "g"},
]}


def test_rules_eligible_via_or_group():
    r = eval_scheme(RULES, {"annual_family_income": 200000, "category": "SC", "disability_percent": None})
    assert r["status"] == "eligible"


def test_rules_unmet_beats_unknown():
    r = eval_scheme(RULES, {"annual_family_income": 300000})
    assert r["status"] == "not_eligible"


def test_rules_needs_info_lists_missing_fields():
    r = eval_scheme(RULES, {"annual_family_income": 100000})
    assert r["status"] == "needs_info" and set(r["missing_fields"]) == {"category", "disability_percent"}


def test_income_boundary_is_inclusive():
    assert eval_scheme(RULES, {"annual_family_income": 250000, "category": "ST"})["status"] == "eligible"
    assert eval_scheme(RULES, {"annual_family_income": 250001, "category": "ST"})["status"] == "not_eligible"


def test_validate_rules_catches_problems():
    bad = {"scheme_id": "X", "scheme_name": "X",
           "criteria": [{"id": "c1", "field": "shoe_size", "op": "~", "source_sids": []}]}
    probs = validate_rules(bad)
    assert len(probs) >= 3


def test_clean_profile_coercion():
    p = clean_profile({"annual_family_income": "2.8 lakh", "category": "obc", "age": "19 years",
                       "is_only_girl_child": "yes", "gender": "alien"})
    assert p["annual_family_income"] == 280000 and p["category"] == "OBC"
    assert p["age"] == 19 and p["is_only_girl_child"] is True and p["gender"] is None


def test_merge_profiles_keeps_old_values():
    m = merge_profiles({"age": 20, "category": "SC"}, {"age": None, "category": "ST"})
    assert m["age"] == 20 and m["category"] == "ST"


def test_number_check():
    prem = "The scholarship amount is Rs 20000 per year for students scoring 60%."
    assert numbers_supported("You get Rs 20,000 per year", prem)
    assert not numbers_supported("You get Rs 25,000 per year", prem)
    assert extract_numbers("Rs 2.5 lakh and 60 per cent") == {"250000", "60"}


def test_rrf_prefers_items_ranked_high_in_both_lists():
    fused = rrf([[1, 2, 3], [2, 1, 4]])
    assert {fused[0][0], fused[1][0]} == {1, 2}


def test_lexical_retrievers_find_the_income_clause():
    chunks = [
        {"chunk_id": "A#c1", "scheme_id": "A", "text": "Eligibility. Annual family income must not exceed Rs 250000.", "sids": ["A:p1:s1"], "marked": ""},
        {"chunk_id": "A#c2", "scheme_id": "A", "text": "Benefits. Tuition fee reimbursement and maintenance allowance.", "sids": ["A:p2:s1"], "marked": ""},
        {"chunk_id": "B#c1", "scheme_id": "B", "text": "Eligibility. Family income limit Rs 100000 for girl students.", "sids": ["B:p1:s1"], "marked": ""},
    ]
    for mode in ("bm25", "tfidf"):
        r = Retriever(chunks, mode=mode)
        assert r.search("income limit", k=1, scheme_id="A")[0]["chunk_id"] == "A#c1"
        assert all(c["scheme_id"] == "B" for c in r.search("income limit", k=3, scheme_id="B"))
