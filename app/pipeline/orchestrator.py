"""Main chatbot flow: preprocess -> intent -> domain guard -> retrieve -> generate."""
from __future__ import annotations

from app.config import Institution, settings
from app.pipeline import domain_guard, web_scraper
from app.pipeline.generator import GroundedGenerator
from app.pipeline.intent_classifier import IntentClassifier
from app.pipeline.preprocessing import preprocess
from app.pipeline.rag import Retriever

_GREETING_TEMPLATE = (
    "Hello! I'm the **{name}** educational assistant. I can help you with "
    "admissions, fees, courses, exams, results, faculty, placements, events, and "
    "campus facilities.\n\nWhat would you like to know?"
)

_MIN_RETRIEVAL_SCORE = 0.08 if settings.lite_mode else 0.30


class InstitutionAssistant:
    """Everything needed to answer for one institution."""

    def __init__(self, inst: Institution) -> None:
        self.inst = inst
        self.intent = IntentClassifier(inst.code)
        self.retriever = Retriever(inst)
        self.generator = GroundedGenerator(inst.code)
        if settings.web_scrape_enabled:
            try:
                web_scraper.refresh_cache(inst, force=False)
                self.retriever._load()  # pick up freshly cached pages if needed
            except Exception:
                pass

    # -- main entrypoint ---------------------------------------------------
    def answer(self, message: str) -> dict:
        text = preprocess(message)
        inst = self.inst

        if domain_guard.is_greeting(text):
            return {
                "reply": _GREETING_TEMPLATE.format(name=inst.full_name),
                "intent": "greeting",
                "confidence": 1.0,
                "in_domain": True,
                "source": "greeting",
                "sources": [],
            }

        intent, confidence = self.intent.predict(text)

        if settings.domain_guard_enabled:
            verdict = domain_guard.assess(
                text, intent, confidence, settings.intent_confidence_threshold, inst
            )
            if not verdict["in_domain"]:
                return {
                    "reply": domain_guard.out_of_domain_message(inst),
                    "intent": "out_of_domain",
                    "confidence": confidence,
                    "in_domain": False,
                    "source": "domain_guard",
                    "sources": [],
                }

        hits = self.retriever.search(text, k=4, intent=intent)
        top = hits[0] if hits else None

        if top and top.get("score", 0) >= _MIN_RETRIEVAL_SCORE:
            context = "\n\n".join(h.get("answer", "") for h in hits[:3] if h.get("answer"))
            reply = None
            if self.generator.ready:
                reply = self.generator.generate(text, context)
            if not reply:
                reply = top.get("answer") or top.get("text", "")
            sources = self._collect_sources(hits)
            return {
                "reply": self._with_footer(reply, inst),
                "intent": intent,
                "confidence": confidence,
                "in_domain": True,
                "source": "rag+generator" if self.generator.ready else "rag",
                "sources": sources,
            }

        # Weak/no retrieval -> safe, scoped fallback (never fabricate specifics).
        return {
            "reply": self._fallback(inst),
            "intent": intent,
            "confidence": confidence,
            "in_domain": True,
            "source": "fallback",
            "sources": [{"title": inst.name, "url": inst.website}],
        }

    # -- helpers -----------------------------------------------------------
    def _collect_sources(self, hits: list[dict]) -> list[dict]:
        seen: set[str] = set()
        out: list[dict] = []
        for h in hits:
            url = h.get("url")
            if url and url not in seen:
                seen.add(url)
                out.append({"title": h.get("title") or url, "url": url})
        return out

    def _with_footer(self, reply: str, inst: Institution) -> str:
        return f"{reply}\n\n_For the latest official details, visit {inst.website}_"

    def _fallback(self, inst: Institution) -> str:
        contact = []
        if inst.contact_email:
            contact.append(f"**{inst.contact_email}**")
        if inst.contact_phone:
            contact.append(f"**{inst.contact_phone}**")
        contact_str = " or ".join(contact) if contact else "the institute office"
        return (
            f"I couldn't find a confirmed answer for that in {inst.name}'s knowledge base. "
            f"For accurate, up-to-date information please check the official website "
            f"{inst.website} or contact {contact_str}."
        )


class Orchestrator:
    """Holds one assistant per institution and routes requests."""

    def __init__(self) -> None:
        self._assistants: dict[str, InstitutionAssistant] = {}

    def get(self, code: str | None) -> InstitutionAssistant:
        inst = settings.institution(code)
        if inst.code not in self._assistants:
            self._assistants[inst.code] = InstitutionAssistant(inst)
        return self._assistants[inst.code]

    def is_built(self, code: str | None) -> bool:
        return settings.institution(code).code in self._assistants

    def chat(self, message: str, institution: str | None = None) -> dict:
        assistant = self.get(institution)
        result = assistant.answer(message)
        result["institution"] = assistant.inst.name
        return result
