"""Simple keyword hints when the classifier is unsure on short queries."""
from __future__ import annotations

from app.pipeline.preprocessing import normalize

_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("contact_info", ("contact", "phone", "email", "call", "reach", "address", "whatsapp")),
    ("course_info", ("course", "courses", "program", "programme", "degree", "bsc", "msc", "bca", "mca")),
    ("admission_query", ("admission", "admit", "apply", "application", "enroll", "eligibility")),
    ("fee_structure", ("fee", "fees", "tuition", "cost", "scholarship")),
    ("exam_schedule", ("exam", "exams", "timetable", "schedule", "paper")),
    ("result_query", ("result", "results", "marks", "grade", "cgpa")),
    ("placement_info", ("placement", "placements", "internship", "recruiter", "job")),
    ("faculty_info", ("faculty", "professor", "teacher", "hod", "department")),
    ("event_info", ("event", "events", "fest", "seminar", "notice")),
    ("infrastructure_info", ("hostel", "campus", "library", "lab", "canteen", "facility")),
]


def hint_intent(text: str) -> str | None:
    norm = normalize(text)
    for intent, words in _HINTS:
        if any(w in norm for w in words):
            return intent
    return None
