"""Stage 4: free text (English / Kannada / mixed) -> structured student profile."""
from __future__ import annotations

import json
import re

from . import llm
from .normalize import parse_amount

FIELDS = {
    "age": "integer, years",
    "gender": "female | male | other",
    "category": "General | SC | ST | OBC",
    "minority_community": "Muslim | Christian | Sikh | Buddhist | Jain | Parsi | none",
    "annual_family_income": "integer, rupees per year (1 lakh = 100000)",
    "domicile_state": "Indian state of residence, e.g. Karnataka",
    "education_level": "class_9_10 | pu_11_12 | diploma | ug | pg | phd | other",
    "course_stream": "free text, e.g. engineering, medicine, arts, science",
    "year_of_study": "integer",
    "previous_marks_percent": "number 0-100 (marks in the last completed exam)",
    "disability_percent": "integer 0-100 (0 only if the user says they have no disability)",
    "institution_type": "government | aided | private",
    "is_only_girl_child": "true | false",
    "parent_occupation": "free text, e.g. farmer, construction worker",
}
ENUMS = {
    "gender": ["female", "male", "other"],
    "category": ["General", "SC", "ST", "OBC"],
    "minority_community": ["Muslim", "Christian", "Sikh", "Buddhist", "Jain", "Parsi", "none"],
    "education_level": ["class_9_10", "pu_11_12", "diploma", "ug", "pg", "phd", "other"],
    "institution_type": ["government", "aided", "private"],
}
INT_FIELDS = {"age", "annual_family_income", "year_of_study", "disability_percent"}
FLOAT_FIELDS = {"previous_marks_percent"}
BOOL_FIELDS = {"is_only_girl_child"}

QUESTIONS = {
    "age": "How old are you?",
    "gender": "What is your gender? (some schemes are only for girl students)",
    "category": "Which category do you belong to: General, SC, ST or OBC?",
    "minority_community": "Do you belong to a notified minority community (Muslim, Christian, Sikh, Buddhist, Jain, Parsi)?",
    "annual_family_income": "What is your family's total annual income, in rupees?",
    "domicile_state": "Which state do you live in?",
    "education_level": "What are you studying now (class 9-10, PU, diploma, UG, PG, PhD)?",
    "course_stream": "Which course or stream are you in (engineering, medicine, arts...)?",
    "year_of_study": "Which year of the course are you in?",
    "previous_marks_percent": "What percentage did you score in your last exam?",
    "disability_percent": "Do you have a disability? If yes, what percentage?",
    "institution_type": "Is your institution government, aided or private?",
    "is_only_girl_child": "Are you the only girl child of your parents?",
    "parent_occupation": "What is your parent's occupation?",
}

_SPEC = "\n".join(f'  "{k}": {v}' for k, v in FIELDS.items())
SYSTEM = f"""You extract a student profile from a message for a scholarship-matching system.
The message may be English, Kannada, Hindi or a mix. Return ONLY JSON:
{{"intent": "find_schemes | ask_about_scheme | out_of_scope",
  "language": "en | kn | hi | other",
  "english_query": "the request restated in one clear English sentence",
  "profile": {{
{_SPEC}
  }}}}
Rules:
- Use null for anything not stated. NEVER guess or assume. Do not infer category from a surname.
- You MAY infer domicile_state from an explicitly named Indian city or district (Mysuru -> Karnataka).
- Convert money to integer rupees: "2.8 lakh" -> 280000.
- intent=ask_about_scheme if the user asks a specific question about a scheme (amount, documents, deadline).
- intent=out_of_scope if the message is unrelated to scholarships, education funding or government schemes.
- "Known so far" holds earlier answers; return only what the NEW message states or corrects (null otherwise).

Example 1
MESSAGE: I'm a 2nd year BE student from Mysuru, OBC, family income is 2.8 lakh. Got 78% in PUC.
OUTPUT: {{"intent":"find_schemes","language":"en","english_query":"Scholarships for a second-year engineering student from Karnataka, OBC, income Rs 2.8 lakh","profile":{{"age":null,"gender":null,"category":"OBC","minority_community":null,"annual_family_income":280000,"domicile_state":"Karnataka","education_level":"ug","course_stream":"engineering","year_of_study":2,"previous_marks_percent":78,"disability_percent":null,"institution_type":null,"is_only_girl_child":null,"parent_occupation":null}}}}

Example 2
MESSAGE: ನಾನು ಎರಡನೇ ವರ್ಷದ ಇಂಜಿನಿಯರಿಂಗ್ ವಿದ್ಯಾರ್ಥಿ, ಕುಟುಂಬದ ವಾರ್ಷಿಕ ಆದಾಯ 2 ಲಕ್ಷ
OUTPUT: {{"intent":"find_schemes","language":"kn","english_query":"Scholarships for a second-year engineering student with family income Rs 2 lakh","profile":{{"age":null,"gender":null,"category":null,"minority_community":null,"annual_family_income":200000,"domicile_state":null,"education_level":"ug","course_stream":"engineering","year_of_study":2,"previous_marks_percent":null,"disability_percent":null,"institution_type":null,"is_only_girl_child":null,"parent_occupation":null}}}}"""


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else None


def clean_profile(raw: dict | None) -> dict:
    """Validate and coerce model output; anything invalid becomes None."""
    raw = raw or {}
    out = {k: None for k in FIELDS}
    for k in FIELDS:
        v = raw.get(k)
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "unknown", "n/a")):
            continue
        if k in INT_FIELDS:
            n = parse_amount(v) if (k == "annual_family_income" and isinstance(v, str)) else None
            n = n if n is not None else _num(v)
            out[k] = int(round(n)) if n is not None else None
        elif k in FLOAT_FIELDS:
            out[k] = _num(v)
        elif k in BOOL_FIELDS:
            out[k] = v if isinstance(v, bool) else str(v).strip().lower() in ("true", "yes", "1")
        elif k in ENUMS:
            out[k] = {e.lower(): e for e in ENUMS[k]}.get(str(v).strip().lower())
        else:
            out[k] = str(v).strip()
    return out


def merge_profiles(prior: dict | None, new: dict) -> dict:
    merged = {k: (prior or {}).get(k) for k in FIELDS}
    for k, v in new.items():
        if v is not None:
            merged[k] = v
    return merged

def _is_document_question(text: str) -> bool:
    """Recognize factual questions about indexed scholarship documents."""
    q = text.lower()
    patterns = [
        r"\b(page|p\.?)\s*\d+\b",
        r"\b(according to|based on|mentioned in|found in|stated in)\b",
        r"\b(source sentences?|source excerpts?|source document|retrieved documents?)\b",
        r"\b(list|enumerate|name)\b.{0,60}\b(schemes?|scholarships?)\b",
        r"\b(schemes?|scholarships?)\b.{0,60}\b(mentioned|listed|found|available|in the documents?)\b",
        r"\b(eligibility criteria|minimum percentage|marks required)\b",
        r"\b(documents required|application deadline|last date|benefits|scholarship amount)\b",
        r"\b(compare|summari[sz]e)\b.{0,80}\b(schemes?|scholarships?)\b",
    ]
    return any(re.search(p, q) for p in patterns)


def extract_profile(text: str, prior: dict | None = None) -> dict:
    user = f"Known so far: {json.dumps(prior or {}, ensure_ascii=False)}\n\nMESSAGE:\n{text}"
    out = llm.complete_json(SYSTEM, user, max_tokens=900)
    new = clean_profile(out.get("profile"))
    if new["annual_family_income"] is None and "income" in text.lower():
        new["annual_family_income"] = parse_amount(text)      # regex safety net
    intent = out.get("intent") if out.get("intent") in (
    "find_schemes", "ask_about_scheme", "out_of_scope"
    ) else "find_schemes"

    # Deterministic override: document questions should bypass eligibility matching.
    if _is_document_question(text):
        intent = "ask_about_scheme"
    return {"intent": intent, "language": out.get("language", "en"),
            "english_query": out.get("english_query") or text,
            "profile": merge_profiles(prior, new), "new_fields": new}
