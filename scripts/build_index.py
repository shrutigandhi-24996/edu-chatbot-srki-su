"""Build a FAISS retrieval index from the KB + scraped web pages.

Usage:
  python scripts/build_index.py --institution SRKI
  python scripts/build_index.py --institution SU --max-docs 60000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.pipeline import web_scraper  # noqa: E402


def collect_documents(code: str, max_docs: int | None) -> list[dict]:
    docs: list[dict] = []
    kb_path = settings.processed_dir / f"{code.lower()}_kb.json"
    if settings.rag_include_dataset and kb_path.exists():
        with open(kb_path, encoding="utf-8") as f:
            rows = json.load(f)
        for r in rows:
            q = (r.get("text") or "").strip()
            a = (r.get("ideal_response") or "").strip()
            if q and a:
                docs.append({"text": q, "answer": a, "intent": r.get("intent", ""), "source": "dataset"})
        if max_docs:
            docs = docs[:max_docs]

    for page in web_scraper.load_cache(code):
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--institution", "-i", default="SRKI", choices=["SRKI", "SU"])
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    code = args.institution.upper()
    docs = collect_documents(code, args.max_docs)
    if not docs:
        print(f"[{code}] no documents found. Run prepare_data.py / scrape_sites.py first.")
        return
    print(f"[{code}] embedding {len(docs)} documents with {settings.embedding_model} ...")

    import faiss
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(settings.embedding_model)
    texts = [d["text"] for d in docs]
    emb = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=args.batch_size,
    )
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    out_dir = settings.index_dir / code.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    with open(out_dir / "documents.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)
    print(f"[{code}] saved index + {len(docs)} docs to {out_dir}")


if __name__ == "__main__":
    main()
