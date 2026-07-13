"""Out-of-domain detection.

Decides whether a query belongs to the educational domain of the served
institution. Combines a domain lexicon with the intent classifier's confidence
so clearly off-topic questions (weather, sports, cooking, celebrities, general
coding help, etc.) get a polite "educational assistant only" redirect instead of
a hallucinated answer.
"""
from __future__ import annotations

from app.config import Institution
from app.pipeline.preprocessing import normalize

# Vocabulary that signals an educational / campus-related query.
EDU_KEYWORDS = {
    "admission", "admissions", "apply", "application", "enroll", "enrolment",
    "eligibility", "fee", "fees", "tuition", "scholarship", "course", "courses",
    "program", "programme", "programs", "degree", "diploma", "bachelor", "master",
    "bsc", "msc", "bca", "mca", "bba", "mba", "bcom", "phd", "syllabus",
    "curriculum", "semester", "sem", "subject", "subjects", "credit", "exam",
    "exams", "examination", "test", "result", "results", "grade", "marks",
    "cgpa", "timetable", "schedule", "faculty", "professor", "teacher", "hod",
    "department", "placement", "placements", "internship", "recruiter", "career",
    "campus", "hostel", "library", "lab", "laboratory", "facility", "facilities",
    "infrastructure", "canteen", "sports", "event", "events", "fest", "seminar",
    "workshop", "notice", "circular", "contact", "email", "phone", "address",
    "location", "college", "university", "institute", "institution", "student",
    "students", "class", "lecture", "attendance", "prospectus", "brochure",
    "accreditation", "naac", "ranking", "research", "convocation", "graduation",
    "transcript", "certificate", "id card", "portal", "login", "erp",
    "counselling", "counseling", "form", "deadline", "entrance", "merit",
}

# Strong off-topic signals that almost never belong to a campus assistant.
OFF_TOPIC_KEYWORDS = {
    "weather", "recipe", "cook", "movie", "song", "lyrics", "cricket score",
    "football", "stock", "bitcoin", "crypto", "horoscope", "joke", "dating",
    "flight", "hotel booking", "restaurant near", "celebrity", "actor",
    "actress", "politics", "election", "war", "medicine dosage", "symptom",
}

GREETINGS = {
    "hi", "hello", "hey", "hii", "helo", "good morning", "good afternoon",
    "good evening", "thanks", "thank you", "bye", "goodbye", "ok", "okay",
}


def is_greeting(text: str) -> bool:
    return normalize(text) in GREETINGS


def is_hard_off_topic(text: str) -> bool:
    """Only the obvious non-educational cases (used when the LLM brain handles
    the softer domain judgement itself)."""
    norm = normalize(text)
    return any(k in norm for k in OFF_TOPIC_KEYWORDS)


def out_of_domain_message(inst: Institution) -> str:
    return (
        f"I'm an **educational assistant for {inst.full_name} ({inst.name})** and its "
        "constituent colleges. I can help with questions about admissions, fees, courses, "
        "exams, results, faculty, placements, events, and campus facilities.\n\n"
        "Your question looks like it's outside that scope, so I can't answer it here. "
        "Please ask me something about the institution or its academic programs."
    )


def assess(
    text: str,
    intent: str,
    confidence: float,
    threshold: float,
    inst: Institution,
) -> dict:
    """Return {"in_domain": bool, "reason": str}."""
    norm = normalize(text)
    tokens = set(norm.split())

    # Institution name / alias mention is a strong in-domain signal.
    mentions_inst = any(alias in norm for alias in inst.aliases) or inst.name.lower() in norm

    has_edu = bool(tokens & EDU_KEYWORDS) or any(k in norm for k in EDU_KEYWORDS if " " in k)
    has_off = any(k in norm for k in OFF_TOPIC_KEYWORDS)

    if mentions_inst or has_edu:
        # Even if off-topic words appear, an educational framing keeps it in-domain.
        return {"in_domain": True, "reason": "edu_signal"}

    if has_off:
        return {"in_domain": False, "reason": "off_topic_keyword"}

    # No educational signal AND the classifier is unsure -> treat as out of domain.
    if confidence < threshold:
        return {"in_domain": False, "reason": "low_confidence_no_edu_signal"}

    return {"in_domain": True, "reason": "confident_intent"}
