"""Turn retrieved web chunks into readable, helpful answers (no LLM required).

When the Groq LLM brain is not configured, this module cleans scraped page text,
extracts contact details, and formats a concise reply instead of dumping raw HTML
extracts on the user.
"""
from __future__ import annotations

import re

from app.config import Institution

# Repeated navigation / chrome on srki.ac.in pages.
_NOISE_PHRASES = (
    "Shree Ramkrishna Institute of Computer Education and Applied Sciences",
    "Contact Us", "Home", "About US", "Social Media", "Students Zone",
    "Photo Gallery", "Video Gallery", "Downloads", "Media College",
    "News/Activities", "Previous-Question-Paper", "E-Magazine",
)

_INTENT_INTROS: dict[str, str] = {
    "admission_query": "Here is what I found about **admissions** at {name}:",
    "fee_structure": "Here is **fee-related** information from the official site:",
    "course_info": "Here are **courses and programs** offered at {name}:",
    "contact_info": "You can reach **{name}** using these official contact details:",
    "exam_schedule": "Here is **examination-related** information:",
    "result_query": "Here is **results-related** information:",
    "placement_info": "Here is **placement / career** information:",
    "faculty_info": "Here is **faculty / department** information:",
    "event_info": "Here are **events and notices**:",
    "infrastructure_info": "Here is **campus / facilities** information:",
}

_PHONE_RE = re.compile(r"\b(?:\+91[- ]?)?(?:72280185\d{2}|722801849[0-9]|0261[- ]?224017[0-9])\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@srki\.ac\.in\b", re.I)
_COURSE_RE = re.compile(
    r"\b(?:B\.?Sc\.?|M\.?Sc\.?|B\.?CA|MCA|BBA|MBA|B\.?Com|Ph\.?D\.?)"
    r"(?:\s*\([^)]+\))?(?:\s+[A-Za-z][A-Za-z &]+)?",
    re.I,
)


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    for phrase in _NOISE_PHRASES:
        text = text.replace(phrase, " ")
    return re.sub(r"\s{2,}", " ", text).strip()


def _extract_contacts(text: str) -> dict[str, list[str]]:
    phones = list(dict.fromkeys(_PHONE_RE.findall(text)))
    emails = list(dict.fromkeys(_EMAIL_RE.findall(text)))
    return {"phones": phones[:6], "emails": emails[:3]}


def _best_snippet(text: str, question: str, max_len: int = 520) -> str:
    """Pick the sentence window most relevant to the question words."""
    text = _clean(text)
    if len(text) <= max_len:
        return text
    q_words = {w for w in re.findall(r"[a-z]{3,}", question.lower())}
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        return text[:max_len].rsplit(" ", 1)[0] + "…"
    scored = []
    for s in sentences:
        if len(s) < 20:
            continue
        words = set(re.findall(r"[a-z]{3,}", s.lower()))
        score = len(words & q_words)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked: list[str] = []
    total = 0
    for _, s in scored[:4]:
        if total + len(s) > max_len:
            break
        picked.append(s)
        total += len(s)
    if not picked:
        return text[:max_len].rsplit(" ", 1)[0] + "…"
    return " ".join(picked)


def _format_contacts(contacts: dict[str, list[str]], inst: Institution) -> str:
    lines: list[str] = []
    if contacts.get("phones"):
        lines.append("- **Phone:** " + ", ".join(contacts["phones"][:4]))
    if contacts.get("emails"):
        lines.append("- **Email:** " + ", ".join(contacts["emails"]))
    if inst.contact_phone and inst.contact_phone not in " ".join(contacts.get("phones", [])):
        lines.append(f"- **Office:** {inst.contact_phone}")
    if inst.contact_email and inst.contact_email not in contacts.get("emails", []):
        lines.append(f"- **Email:** {inst.contact_email}")
    lines.append(f"- **Website:** {inst.website}")
    return "\n".join(lines)


def _format_courses(text: str) -> str | None:
    found = list(dict.fromkeys(m.group(0).strip() for m in _COURSE_RE.finditer(text)))
    if len(found) < 3:
        return None
    bullets = "\n".join(f"- {c}" for c in found[:14])
    extra = f"\n- _…and more on the official courses page._" if len(found) > 14 else ""
    return bullets + extra


def compose(
    question: str,
    intent: str,
    hits: list[dict],
    inst: Institution,
) -> str:
    """Build a user-friendly markdown answer from retrieval hits."""
    if not hits:
        return ""

    combined = " ".join(h.get("answer") or h.get("text") or "" for h in hits[:3])
    contacts = _extract_contacts(combined)
    intro = _INTENT_INTROS.get(intent, "Here is what I found on the official site:").format(
        name=inst.name
    )

    # Contact intent → lead with structured contact block
    if intent == "contact_info" or "contact" in question.lower():
        body = _format_contacts(contacts, inst)
        return f"{intro}\n\n{body}"

    # Course intent → try bullet list of programmes
    if intent in {"course_info", "admission_query"}:
        courses = _format_courses(combined)
        if courses:
            snippet = _best_snippet(combined, question, max_len=280)
            extra = f"\n\n{snippet}" if snippet and len(snippet) > 40 else ""
            return f"{intro}\n\n{courses}{extra}"

    snippet = _best_snippet(combined, question)
    parts = [intro, "", snippet]

    # Append contacts when useful and not already in snippet
    if intent in {"admission_query", "fee_structure", "contact_info"}:
        if contacts["phones"] or contacts["emails"]:
            parts += ["", "**Need more help?**"]
            if contacts["phones"]:
                parts.append(f"- Call: {', '.join(contacts['phones'][:3])}")
            if contacts["emails"]:
                parts.append(f"- Email: {contacts['emails'][0]}")

    parts += [""]
    return "\n".join(parts)
