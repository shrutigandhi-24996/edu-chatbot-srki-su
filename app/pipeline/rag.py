"""Retrieval-augmented knowledge base.

Documents come from two sources:
  1. Dataset B rows  -> (text question, ideal_response) grounded QA pairs.
  2. Scraped official web pages -> chunked page text with source URLs.

A FAISS index built by ``scripts/build_index.py`` is loaded when present. If it
is missing, a capped in-memory index is built on the fly so the bot still works.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import Institution, settings
from app.pipeline import web_scraper

_FALLBACK_MAX_DOCS = 8000  # cap for on-the-fly index (keeps CPU build fast)


class Retriever:
    def __init__(self, inst: Institution) -> None:
        self.inst = inst
        self.code = inst.code
        self.documents: list[dict] = []
        self._index = None
        self._embedder = None
        self._matrix = None  # numpy fallback matrix when FAISS index absent
        self._tfidf = None  # (vectorizer, matrix) for lite mode
        self.backend = "empty"
        self._load()

    # -- loading -----------------------------------------------------------
    def _index_dir(self) -> Path:
        return settings.index_dir / self.code.lower()

    def _load(self) -> None:
        if settings.lite_mode:
            self._build_tfidf()
            return
        if self._load_faiss():
            return
        if not self._build_in_memory():
            # sentence-transformers unavailable -> lightweight fallback
            self._build_tfidf()

    def _get_embedder(self):
        if self._embedder is None:
            from app.pipeline.embedding import get_shared_embedder

            self._embedder = get_shared_embedder()
        return self._embedder

    def _load_faiss(self) -> bool:
        idx_dir = self._index_dir()
        idx_path = idx_dir / "faiss.index"
        docs_path = idx_dir / "documents.json"
        if not (idx_path.exists() and docs_path.exists()):
            return False
        try:
            import faiss

            self._index = faiss.read_index(str(idx_path))
            with open(docs_path, encoding="utf-8") as f:
                self.documents = json.load(f)
            self._get_embedder()
            self.backend = f"faiss ({len(self.documents)} docs)"
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[rag] FAISS load failed: {exc}")
            return False

    def _build_in_memory(self) -> bool:
        docs = self._collect_documents(limit=_FALLBACK_MAX_DOCS)
        if not docs:
            self.backend = "empty (run scripts/build_index.py)"
            return False
        try:
            import numpy as np

            embedder = self._get_embedder()
            texts = [d["text"] for d in docs]
            emb = embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
            )
            self._matrix = np.asarray(emb, dtype="float32")
            self.documents = docs
            self.backend = f"in-memory ({len(docs)} docs)"
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[rag] in-memory build failed: {exc}")
            return False

    def _build_tfidf(self) -> None:
        """Lightweight TF-IDF retrieval index (no PyTorch / FAISS)."""
        docs = self._collect_documents(limit=_FALLBACK_MAX_DOCS)
        if not docs:
            self.backend = "empty (scrape a site first)"
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            matrix = normalize(vectorizer.fit_transform([d["text"] for d in docs]))
            self._tfidf = (vectorizer, matrix)
            self.documents = docs
            self.backend = f"tfidf-lite ({len(docs)} docs)"
        except Exception as exc:  # pragma: no cover
            print(f"[rag] tfidf build failed: {exc}")
            self.backend = "empty"

    def _collect_documents(self, limit: int | None = None) -> list[dict]:
        docs: list[dict] = []
        # Dataset B QA pairs (processed) - only when explicitly enabled, because
        # the templated `ideal_response` values are not reliable factual answers.
        kb_path = settings.processed_dir / f"{self.code.lower()}_kb.json"
        if settings.rag_include_dataset and kb_path.exists():
            with open(kb_path, encoding="utf-8") as f:
                rows = json.load(f)
            for r in rows:
                q = (r.get("text") or "").strip()
                a = (r.get("ideal_response") or "").strip()
                if q and a:
                    docs.append(
                        {
                            "text": q,
                            "answer": a,
                            "intent": r.get("intent", ""),
                            "source": "dataset",
                        }
                    )
                    if limit and len(docs) >= limit:
                        break
        # Scraped web chunks
        for page in web_scraper.load_cache(self.code):
            for chunk in page.get("chunks", []):
                docs.append(
                    {
                        "text": chunk,
                        "answer": chunk,
                        "intent": "",
                        "source": "web",
                        "url": page.get("url"),
                        "title": page.get("title"),
                    }
                )
        return docs

    @property
    def ready(self) -> bool:
        return (
            self._index is not None
            or self._matrix is not None
            or self._tfidf is not None
        )

    # -- search ------------------------------------------------------------
    def search(self, query: str, k: int = 4, intent: str | None = None) -> list[dict]:
        if not self.ready or not query.strip():
            return []
        import numpy as np

        if self._tfidf is not None:
            from sklearn.preprocessing import normalize

            vectorizer, matrix = self._tfidf
            qv = normalize(vectorizer.transform([query]))
            sims = (matrix @ qv.T).toarray().ravel()
            order = np.argsort(sims)[::-1][: k * 3]
            candidates = [{**self.documents[i], "score": float(sims[i])} for i in order]
        else:
            vec = self._get_embedder().encode([query], normalize_embeddings=True)
            vec = np.asarray(vec, dtype="float32")
            if self._index is not None:
                scores, idxs = self._index.search(vec, k * 3)
                candidates = [
                    {**self.documents[i], "score": float(s)}
                    for s, i in zip(scores[0], idxs[0])
                    if 0 <= i < len(self.documents)
                ]
            else:
                sims = (self._matrix @ vec[0])
                order = np.argsort(sims)[::-1][: k * 3]
                candidates = [{**self.documents[i], "score": float(sims[i])} for i in order]

        # Rank by semantic score with a boost for real web content, because the
        # dataset's templated ideal_response answers are not reliably factual.
        for c in candidates:
            boost = 0.08 if c.get("source") == "web" else 0.0
            if intent and c.get("intent") == intent:
                boost += 0.02
            c["_rank"] = c.get("score", 0.0) + boost
        candidates.sort(key=lambda c: c["_rank"], reverse=True)
        return candidates[:k]
