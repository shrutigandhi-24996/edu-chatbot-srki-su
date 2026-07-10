# Deployment Guide — getting a public link

You do **not** need a local GPU. Train on Colab (free GPU), push models to the
Hugging Face Hub, then deploy the app to a free host that gives you a public URL.

## Recommended: Hugging Face Spaces (Docker) — free public URL

1. Train models with the notebooks in `notebooks/` and push them to the Hub
   (`PUSH_TO_HUB = True`). You'll get repos like `you/srki-intent-roberta`.
2. Create a new Space → **SDK: Docker** → link/upload this repository.
3. Copy `deploy/README_HF_SPACE.md` to the repo root as `README.md`
   (its YAML front-matter configures the Space).
4. In the Space **Settings → Variables and secrets**, add:
   - `SRKI_INTENT_MODEL = you/srki-intent-roberta`
   - `SRKI_GENERATOR_MODEL = you/srki-generator`  (optional)
   - `USE_GENERATOR = true`  (only if you trained a generator)
5. The Space builds the Docker image and starts the app. Your public link is:
   `https://<your-user>-<space-name>.hf.space`

The chat UI is at `/`, health at `/api/health`, and the API at `/api/chat`.

## Alternative: Render (Docker, free tier)

- Push this repo to GitHub.
- On Render: **New → Web Service → Build from Dockerfile**. `render.yaml` is
  already provided. Add the same model env vars. Render gives you an
  `https://edu-chatbot.onrender.com`-style URL.

## Alternative: quick temporary public link from your PC

For a short-lived demo without cloud deploy:
```bash
python run.py                       # starts the app on http://127.0.0.1:8000
# in another terminal (needs a tunneler such as cloudflared / ngrok):
cloudflared tunnel --url http://localhost:8000
```
This prints a temporary public `https://...trycloudflare.com` URL.

## Notes
- Without configured model repos, the app still runs using the embedding-fallback
  intent classifier and dataset/web retrieval — so a demo works immediately.
- Large data (`data/raw`, FAISS indexes) is git-ignored; the app rebuilds a
  capped in-memory index from `data/processed/*_kb.json` and the web cache.
