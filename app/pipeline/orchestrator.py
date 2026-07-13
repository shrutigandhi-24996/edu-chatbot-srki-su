"""Main chatbot flow: preprocess -> intent -> domain guard -> retrieve -> generate."""
from __future__ import annotations

from app.config import Institution, settings
from app.pipeline import domain_guard, web_scraper
from app.pipeline.answer_composer import compose as compose_answer
from app.pipeline.generator import GroundedGenerator
from app.pipeline.intent_hints import hint_intent
from app.pipeline.intent_classifier import IntentClassifier
from app.pipeline.llm_brain import LLMBrain
from app.pipeline.preprocessing import preprocess
from app.pipeline.rag import Retriever

_GREETING_TEMPLATE = (
    "Hello! I'm the **{name}** educational assistant.\n\n"
    "I can help you with:\n"
    "- Admissions and eligibility\n"
    "- Courses and programs\n"
    "- Fees and scholarships\n"
    "- Exams, results, and timetables\n"
    "- Placements, faculty, and campus facilities\n"
    "- Contact details\n\n"
    "Try asking: *\"What courses are offered?\"* or *\"How do I apply for admission?\"*"
)

_INTENT_FALLBACKS: dict[str, str] = {
    "admission_query": (
        "For **admission** details, visit the Admission Corner on the official website "
        "or contact the admission office."
    ),
    "fee_structure": (
        "For the latest **fee structure**, check the Fees page on the official website "
        "or email the institute office."
    ),
    "course_info": (
        "SRKI offers UG and PG programs in Computer Science, IT, Biotechnology, "
        "Microbiology, Chemistry, and more. See the full list on the Courses Offered page."
    ),
    "contact_info": (
        "Contact SRKI: **info@srki.ac.in**, phone **7228018497**, or visit **srki.ac.in/contact**."
    ),
    "exam_schedule": (
        "Exam timetables and previous question papers are published on the SRKI website "
        "under Examination / Previous Question Paper sections."
    ),
    "placement_info": (
        "Placement activities are coordinated through Sarvajanik University. "
        "Contact the institute office for the latest placement drives."
    ),
}

_MIN_RETRIEVAL_SCORE = 0.08 if settings.lite_mode else 0.30


class InstitutionAssistant:
    """Everything needed to answer for one institution."""

    def __init__(self, inst: Institution) -> None:
        self.inst = inst
        self.intent = IntentClassifier(inst.code)
        self.retriever = Retriever(inst)
        self.generator = GroundedGenerator(inst.code)
        self.brain = LLMBrain(inst)
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
        hinted = hint_intent(text)
        if hinted and (confidence < settings.intent_confidence_threshold or intent != hinted):
            # Prefer obvious keyword intent for common user questions.
            if hinted in self.intent.labels or not self.intent.labels:
                intent, confidence = hinted, max(confidence, 0.55)

        # Domain gating. With the LLM brain we only hard-block obvious off-topic
        # queries and let the model itself decline softer non-educational ones,
        # so it can still answer general educational questions. Without the brain
        # we use the stricter keyword+confidence guard.
        if settings.domain_guard_enabled:
            if self.brain.ready:
                if domain_guard.is_hard_off_topic(text):
                    return self._ood(inst, confidence)
            else:
                verdict = domain_guard.assess(
                    text, intent, confidence, settings.intent_confidence_threshold, inst
                )
                if not verdict["in_domain"]:
                    return self._ood(inst, confidence)

        hits = self.retriever.search(text, k=4, intent=intent)
        top = hits[0] if hits else None
        if not top or top.get("score", 0) < _MIN_RETRIEVAL_SCORE:
            page_hits = self.retriever.pages_for_intent(intent)
            if page_hits:
                hits = page_hits + hits
                top = hits[0]

        # LLM brain: write a helpful, grounded answer from retrieved context.
        if self.brain.ready:
            context = "\n\n".join(
                h.get("answer", "") for h in hits[:4] if h.get("answer")
            )
            reply = self.brain.answer(text, context)
            if reply:
                return {
                    "reply": reply,
                    "intent": intent,
                    "confidence": confidence,
                    "in_domain": True,
                    "source": "llm+web" if context else "llm",
                    "sources": self._collect_sources(hits) if context else [],
                }

        if top and top.get("score", 0) >= _MIN_RETRIEVAL_SCORE:
            context = "\n\n".join(h.get("answer", "") for h in hits[:3] if h.get("answer"))
            reply = None
            if self.generator.ready:
                reply = self.generator.generate(text, context)
            if not reply:
                reply = compose_answer(text, intent, hits, inst)
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

        # Intent-specific helpful fallback before the generic message.
        if intent in _INTENT_FALLBACKS:
            return {
                "reply": self._with_footer(_INTENT_FALLBACKS[intent], inst),
                "intent": intent,
                "confidence": confidence,
                "in_domain": True,
                "source": "intent_guide",
                "sources": [{"title": inst.name, "url": inst.website}],
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
    def _ood(self, inst: Institution, confidence: float) -> dict:
        return {
            "reply": domain_guard.out_of_domain_message(inst),
            "intent": "out_of_domain",
            "confidence": confidence,
            "in_domain": False,
            "source": "domain_guard",
            "sources": [],
        }

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
