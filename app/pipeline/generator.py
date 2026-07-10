"""Grounded answer generation with T5 / FLAN-T5.

The generator is prompted to answer *only* from the retrieved context to keep it
grounded and reduce hallucination. It is optional: when ``USE_GENERATOR`` is
false (or the model can't be loaded) the orchestrator falls back to returning the
best retrieved ``ideal_response`` directly.
"""
from __future__ import annotations

from app.config import settings


class GroundedGenerator:
    def __init__(self, institution_code: str) -> None:
        self.code = institution_code.upper()
        self.ready = False
        self._tokenizer = None
        self._model = None
        if settings.use_generator:
            self._load()

    def _load(self) -> None:
        source = settings.generator_source(self.code)
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(source)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(source)
            self._model.eval()
            self.ready = True
            print(f"[generator] loaded {source}")
        except Exception as exc:  # pragma: no cover
            print(f"[generator] load failed ({source}): {exc}")
            self.ready = False

    def generate(self, question: str, context: str) -> str | None:
        if not self.ready:
            return None
        context = (context or "").strip()[: settings.generator_max_input_chars]
        if not context:
            return None
        import torch

        prompt = (
            "You are an educational assistant. Answer the question using ONLY the "
            "context below. If the answer is not in the context, say you don't have "
            "that information.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        )
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=settings.generator_max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        text = self._tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return text or None
