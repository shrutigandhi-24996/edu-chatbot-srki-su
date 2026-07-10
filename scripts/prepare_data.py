"""Prepare training + serving artifacts from the raw datasets.

For an institution (SRKI or SU) this reads:
  Dataset A: id, text, intent                          -> intent classifier data
  Dataset B: id, text, intent, dialogue_act, context, ideal_response
                                                        -> RAG KB + generator data

Outputs (data/processed/):
  <code>_train.json / _val.json / _test.json   stratified splits for classifiers
  <code>_label_map.json                        {"labels": [...]}
  <code>_intent_examples.json                  few examples/intent for embedding fallback
  <code>_kb.json                               grounded QA pairs for retrieval
  <code>_gen.json                              (input_text, target_response) for T5/FLAN-T5

Usage:
  python scripts/prepare_data.py --institution SRKI
  python scripts/prepare_data.py --institution SU --examples-per-intent 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402

OUT = settings.processed_dir


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, on_bad_lines="skip")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _safe_split(records, labels, test_size):
    """Stratified split that gracefully degrades when classes are too small."""
    from collections import Counter

    min_count = min(Counter(labels).values()) if labels else 0
    stratify = labels if min_count >= 2 else None
    return train_test_split(
        records, labels, test_size=test_size, random_state=42, stratify=stratify
    )


def _three_way_split(records, labels, test_size, val_size):
    """Split into train/val/test robustly for small, imbalanced label sets."""
    trainval, test, y_trainval, _ = _safe_split(records, labels, test_size)
    rel_val = val_size / (1 - test_size)
    train, val, _, _ = _safe_split(trainval, y_trainval, rel_val)
    return train, val, test


def _write_json(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  wrote {name} ({len(data)} records)")


def prepare(code: str, examples_per_intent: int, test_size: float, val_size: float) -> None:
    code = code.upper()
    path_a, path_b = settings.dataset_paths(code)
    print(f"[{code}] Dataset A: {path_a}")
    print(f"[{code}] Dataset B: {path_b}")

    # ---- Dataset A -> classifier splits ----
    df_a = _read_csv(path_a)
    df_a = df_a[(df_a["text"].str.len() > 0) & (df_a["intent"].str.len() > 0)]
    df_a = df_a.drop_duplicates(subset=["text"])
    labels = sorted(df_a["intent"].unique().tolist())
    print(f"[{code}] {len(df_a)} rows, {len(labels)} intents")

    records = df_a[["text", "intent"]].to_dict("records")
    y = df_a["intent"].tolist()
    train, val, test = _three_way_split(records, y, test_size, val_size)

    lc = code.lower()
    _write_json(f"{lc}_train.json", train)
    _write_json(f"{lc}_val.json", val)
    _write_json(f"{lc}_test.json", test)
    _write_json(f"{lc}_label_map.json", {"labels": labels})

    # few examples per intent (for the zero-training embedding fallback)
    examples: dict[str, list[str]] = {}
    for intent, grp in df_a.groupby("intent"):
        texts = grp["text"].tolist()[:examples_per_intent]
        examples[intent] = texts
    _write_json(f"{lc}_intent_examples.json", examples)

    # ---- Dataset B -> KB + generator data ----
    df_b = _read_csv(path_b)
    if "ideal_response" in df_b.columns:
        df_b = df_b[
            (df_b["text"].str.len() > 0) & (df_b["ideal_response"].str.len() > 0)
        ]
        df_b = df_b.drop_duplicates(subset=["text"])
        kb = df_b[
            [c for c in ["text", "intent", "context", "dialogue_act", "ideal_response"]
             if c in df_b.columns]
        ].to_dict("records")
        _write_json(f"{lc}_kb.json", kb)

        gen = [
            {"input_text": r["text"], "target_text": r["ideal_response"]}
            for r in kb
        ]
        _write_json(f"{lc}_gen.json", gen)
    else:
        print(f"[{code}] Dataset B has no ideal_response column; skipping KB/gen.")

    print(f"[{code}] done.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare chatbot datasets")
    ap.add_argument("--institution", "-i", default="SRKI", choices=["SRKI", "SU"])
    ap.add_argument("--examples-per-intent", type=int, default=50)
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--val-size", type=float, default=0.10)
    args = ap.parse_args()
    prepare(args.institution, args.examples_per_intent, args.test_size, args.val_size)


if __name__ == "__main__":
    main()
