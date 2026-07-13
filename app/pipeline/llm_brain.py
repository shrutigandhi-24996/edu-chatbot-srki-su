"""LLM answer brain (Groq, OpenAI-compatible chat API).

Writes clear, helpful answers grounded on the scraped official-site content.
Runs as simple HTTPS calls, so it needs no GPU and negligible RAM (ideal for
free hosts). Enabled only when ``GROQ_API_KEY`` is configured.
"""
from __future__ import annotations

import httpx

from app.config import Institution, settings

SYSTEM_TEMPLATE = (
    "You are a friendly, knowledgeable educational assistant for {full_name} "
    "({name}) and its constituent colleges.\n"
    "\n"
    "What you help with:\n"
    "- Questions about {name}: admissions, fees, courses/programs, exams, results, "
    "faculty, placements, events, campus facilities, contact details.\n"
    "- General educational and academic guidance (study help, career/course advice, "
    "explaining academic concepts).\n"
    "\n"
    "How to answer:\n"
    "- Prefer the OFFICIAL CONTEXT below (from the institution's website). If it "
    "answers the question, be specific and use it.\n"
    "- If the context does NOT contain the specific fact (e.g. exact fee, date, "
    "phone number), do not invent it. Give general helpful guidance and suggest "
    "checking {website} or contacting {contact}.\n"
    "- Be concise, warm, and use markdown (short paragraphs, bullet points) when "
    "useful.\n"
    "\n"
    "Scope rule: If the question is clearly NOT about education, academics, careers, "
    "or this institution (e.g. weather, sports, politics, cooking, entertainment), "
    "politely reply that you are an educational assistant for {name} and can only "
    "help with educational questions. Do not answer the off-topic question."
)


class LLMBrain:
    def __init__(self, inst: Institution) -> None:
        self.inst = inst
        self.enabled = settings.llm_brain_enabled()

    @property
    def ready(self) -> bool:
        return self.enabled

    def _contact(self) -> str:
        parts = []
        if self.inst.contact_email:
            parts.append(self.inst.contact_email)
        if self.inst.contact_phone:
            parts.append(self.inst.contact_phone)
        return " / ".join(parts) if parts else "the institute office"

    def answer(
        self,
        question: str,
        context: str = "",
        history: list[dict] | None = None,
    ) -> str | None:
        if not self.enabled:
            return None
        system = SYSTEM_TEMPLATE.format(
            full_name=self.inst.full_name,
            name=self.inst.name,
            website=self.inst.website,
            contact=self._contact(),
        )
        context = (context or "").strip()
        user = (
            (f"OFFICIAL CONTEXT:\n{context}\n\n" if context else "")
            + f"QUESTION: {question}"
        )
        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user})

        try:
            resp = httpx.post(
                f"{settings.groq_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json={
                    "model": settings.groq_model,
                    "messages": messages,
                    "temperature": settings.llm_temperature,
                    "max_tokens": settings.llm_max_tokens,
                },
                timeout=settings.llm_request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return (data["choices"][0]["message"]["content"] or "").strip() or None
        except Exception as exc:  # pragma: no cover
            print(f"[llm_brain] request failed: {exc}")
            return None
