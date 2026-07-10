---
title: Educational Chatbot
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🎓 Innovative Educational Chatbot (SRKI · SU)

A hybrid, domain-aware educational assistant for **Shree Ramkrishna Institute of
Computer Education and Applied Sciences (SRKI)** and **Sarvajanik University (SU)**
and its constituent colleges. It answers admissions, fees, courses, exams,
results, faculty, placements, events and campus questions — and politely refuses
anything outside the educational domain.

## Architecture

```
User query
   │
   ▼
Preprocess ──▶ Intent classifier            ──▶ Domain guard (in/out of domain)
              (BERT / DistilBERT / RoBERTa,       │
               embedding fallback)                │ out-of-domain → polite redirect
                                                  ▼ in-domain
                                     RAG retriever (FAISS over Dataset B
                                     ideal_response + scraped official web)
                                                  │
                                                  ▼
                                     T5 / FLAN-T5 grounded generator
                                     (optional; else best retrieved answer)
                                                  │
                                                  ▼
                                        Answer + sources + intent
```

- **Intent understanding** — fine-tuned transformer encoders (BERT, DistilBERT,
  RoBERTa) compared in a notebook; the best one is loaded at runtime. Until you
  train them, a zero-training **embedding classifier** keeps the bot working.
- **Answer generation** — grounded **T5 / FLAN-T5** over retrieved context to
  reduce hallucination (optional, toggled by `USE_GENERATOR`).
- **Knowledge** — Dataset B `ideal_response` pairs + **live scrapers** for
  `srki.ac.in` and `sarvajanikuniversity.ac.in`.
- **Out-of-domain guard** — non-educational questions get a scoped, honest reply.

## Datasets

| File | Rows | Fields | Used for |
|------|------|--------|----------|
| `Dataset_A_SRKI.csv` | 22,000 | `text, intent` | intent classifiers |
| `Dataset_B_SRKI.csv` | 22,000 | `+ dialogue_act, context, ideal_response` | RAG + generator |
| `SU_final_250k_A.csv` | 250,000 | `text, intent` | intent classifiers (SU) |
| `SU_final_250k_B.csv` | 250,000 | `+ dialogue_act, context, ideal_response` | RAG + generator (SU) |

## ⚠️ Important note about your datasets

After inspecting the data:

- **Dataset A (`text, intent`) is clean and great for intent classification** —
  11 balanced SRKI intents (~235 unique templated questions repeated to 22k rows),
  ~20 SU intents over 250k rows.
- **Dataset B's `ideal_response` is templated filler that does NOT answer the
  questions** (e.g. *"What is the contact number?"* → *"The admission for MSc IT
  usually starts in June."*). It cannot be used to ground factual replies, and a
  T5/FLAN-T5 model trained directly on `text → ideal_response` would learn to
  produce similar nonsense.

**Consequences / design choices:**
1. Factual answers are grounded on **real scraped content** from the official
   websites (web-first RAG). Dataset B answers are excluded from retrieval by
   default (`RAG_INCLUDE_DATASET=false`).
2. Use Dataset B mainly for **dialogue-act / conversational-style** learning, or
   regenerate a proper `ideal_response` (e.g. with the scraped content or an LLM)
   before fine-tuning the generator for factual QA.

## Quick start (local, no GPU needed)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy config.env.example .env      # then edit dataset paths if needed

# 1) Build serving artifacts from your CSVs
python scripts/prepare_data.py --institution SRKI

# 2) (optional) scrape official site + build FAISS index
python scripts/scrape_sites.py --institution SRKI
python scripts/build_index.py  --institution SRKI

# 3) run
python run.py                     # http://127.0.0.1:8000
```

The app runs immediately using the **embedding-fallback** classifier. Train the
transformers for higher accuracy (next section).

## Training the models (on free GPU — not your local machine)

Open the notebooks on **Google Colab / Kaggle** (GPU runtime):

- `notebooks/01_train_intent_classifiers.ipynb` — trains & compares
  **BERT / DistilBERT / RoBERTa**, reports accuracy/precision/recall/F1, saves
  the best model and can push it to the Hugging Face Hub.
- `notebooks/02_train_t5_generator.ipynb` — fine-tunes **T5 / FLAN-T5** on
  `text → ideal_response`.

After pushing to the Hub, point the app at them in `.env`:

```env
SRKI_INTENT_MODEL=your-user/srki-intent-roberta
SRKI_GENERATOR_MODEL=your-user/srki-generator
USE_GENERATOR=true
```

For SU, repeat with `--institution SU` and the SU CSVs (subsample the 250k rows
for a fast first run using `MAX_ROWS` in the notebook).

## Deploy for a public link

See **`deploy/DEPLOYMENT.md`**. Recommended: **Hugging Face Spaces (Docker)** —
free, gives a `https://<user>-<space>.hf.space` URL and loads your trained models
from the Hub.

## API

- `GET  /api/health` — backend status, intent backend, labels
- `GET  /api/institutions` — available institutions
- `POST /api/chat` — `{ "message": "...", "session_id": "...", "institution": "SRKI" }`
- `POST /api/web/refresh?institution=SRKI` — re-scrape official pages

## Project layout

```
app/            FastAPI backend + pipeline (intent, rag, generator, domain guard)
scripts/        prepare_data, scrape_sites, build_index
notebooks/      Colab training notebooks (classifiers + generator)
frontend/       chat UI (HTML/CSS/JS)
deploy/         deployment guides + HF Space config
data/           processed artifacts, indexes, web cache (git-ignored)
```

## Roadmap
- [x] SRKI end-to-end (intent → guard → RAG → answer)
- [x] SU wired into the same pipeline (switch with `ACTIVE_INSTITUTION=SU`)
- [ ] Train & host transformer + generator models on the Hub
- [ ] Deployed public Space
