---
title: Educational Chatbot
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Educational Chatbot (SRKI / SU)

Hybrid educational assistant: fine-tuned intent classifier (BERT/DistilBERT/RoBERTa)
+ RAG retrieval + T5/FLAN-T5 grounded generation, with out-of-domain guarding.

## How this Space works
- The FastAPI app serves the chat UI at `/` and the API at `/api/chat`.
- Trained models are pulled from the Hugging Face Hub at startup. Set them as
  **Space secrets / variables**:
  - `SRKI_INTENT_MODEL=<your-user>/srki-intent-roberta`
  - `SRKI_GENERATOR_MODEL=<your-user>/srki-generator` and `USE_GENERATOR=true`
- If no model repos are configured, the app runs with the embedding-fallback
  classifier so the Space still works.

> This file's YAML front-matter is required by Hugging Face Spaces. When you
> create the Space, copy it to the repo root as `README.md`.
