"""Intent recognition.

Prefers a fine-tuned transformer classifier (BERT / DistilBERT / RoBERTa) loaded
from a local directory or the Hugging Face Hub. When none is available it falls
back to a zero-training embedding classifier built from a handful of labelled
example utterances, so the chatbot is usable immediately after data prep.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings

_LABEL_FILE = "{code}_label_map.json"
_EXAMPLES_FILE = "{code}_intent_examples.json"


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class IntentClassifier:
    """Predict the intent of an educational query with a confidence score."""

    def __init__(self, institution_code: str | None = None) -> None:
        self.code = (institution_code or settings.active_institution).upper()
        self.backend = "uninitialized"
        self.labels: list[str] = []
        self._pipe = None  # transformers pipeline
        self._embedder = None
        self._centroids = None  # (labels, matrix)
        self._load()

    # -- loading -----------------------------------------------------------
    def _load(self) -> None:
        source = settings.intent_model_source(self.code)
        if source and self._load_transformer(source):
            return
        self._load_embedding_fallback()

    def _load_transformer(self, source: str) -> bool:
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(source)
            model = AutoModelForSequenceClassification.from_pretrained(source)
            self._pipe = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                truncation=True,
                max_length=settings.max_seq_length,
                top_k=None,
            )
            id2label = model.config.id2label
            self.labels = [id2label[i] for i in sorted(id2label)]
            self.backend = f"transformer:{Path(source).name or source}"
            return True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[intent] transformer load failed ({source}): {exc}")
            return False

    def _load_embedding_fallback(self) -> None:
        label_map = _load_json(settings.processed_dir / _LABEL_FILE.format(code=self.code.lower()))
        examples = _load_json(
            settings.processed_dir / _EXAMPLES_FILE.format(code=self.code.lower())
        )
        if label_map:
            self.labels = label_map.get("labels", [])
        if not examples:
            self.backend = "none (run scripts/prepare_data.py)"
            return
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(settings.embedding_model)
            labels: list[str] = []
            vectors = []
            for intent, texts in examples.items():
                if not texts:
                    continue
                emb = self._embedder.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                )
                vectors.append(np.mean(emb, axis=0))
                labels.append(intent)
            matrix = np.vstack(vectors)
            # renormalize centroids
            matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
            self._centroids = (labels, matrix)
            if not self.labels:
                self.labels = labels
            self.backend = "embedding-fallback"
        except Exception as exc:  # pragma: no cover
            print(f"[intent] embedding fallback failed: {exc}")
            self.backend = "none"

    @property
    def ready(self) -> bool:
        return self._pipe is not None or self._centroids is not None

    # -- prediction --------------------------------------------------------
    def predict(self, text: str) -> tuple[str, float]:
        results = self.predict_scores(text)
        if not results:
            return "unknown", 0.0
        label, score = results[0]
        return label, score

    def predict_scores(self, text: str) -> list[tuple[str, float]]:
        """Return (label, score) pairs sorted by score, best first."""
        text = (text or "").strip()
        if not text:
            return []
        if self._pipe is not None:
            return self._predict_transformer(text)
        if self._centroids is not None:
            return self._predict_embedding(text)
        return []

    def _predict_transformer(self, text: str) -> list[tuple[str, float]]:
        out = self._pipe(text)
        # pipeline with top_k=None returns list[list[dict]] or list[dict]
        scores = out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
        pairs = [(d["label"], float(d["score"])) for d in scores]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def _predict_embedding(self, text: str) -> list[tuple[str, float]]:
        import numpy as np

        labels, matrix = self._centroids
        vec = self._embedder.encode([text], normalize_embeddings=True)[0]
        sims = matrix @ vec  # cosine (both normalized)
        order = np.argsort(sims)[::-1]
        return [(labels[i], float(sims[i])) for i in order]
