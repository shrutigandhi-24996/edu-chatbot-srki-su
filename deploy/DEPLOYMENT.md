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

## Render (Docker, free tier) — step by step

Render's free tier has **512 MB RAM**, which is too small for PyTorch. This repo
therefore ships a **lightweight TF-IDF mode** (`LITE_MODE=true`, no PyTorch) used
by `deploy/Dockerfile.render`. Answer quality is a bit lower than the full
embedding/transformer mode, but it runs comfortably in free-tier RAM.

Steps (repo already on GitHub at `shrutigandhi-24996/edu-chatbot-srki-su`):

1. Go to <https://dashboard.render.com> and sign in with **GitHub**.
2. Click **New → Blueprint** (recommended). Render reads `render.yaml`
   automatically.  *(Or **New → Web Service → Build from a Git repository** and
   set Docker + `dockerfilePath: deploy/Dockerfile.render`.)*
3. Select the repository **edu-chatbot-srki-su** and approve access.
4. Confirm the plan is **Free** and click **Apply / Create**.
5. Wait for the first build + deploy (a few minutes). Your public URL will look
   like `https://edu-chatbot.onrender.com`.

One-click blueprint link:
`https://render.com/deploy?repo=https://github.com/shrutigandhi-24996/edu-chatbot-srki-su`

Notes:
- Free instances **sleep after ~15 min idle**; the first request after sleeping
  takes ~30–60 s to wake and warm up.
- The app scrapes the official site on startup (`WEB_MAX_PAGES=25`) and builds a
  TF-IDF index in memory — no model files needed.
- For **full quality** (embeddings + your trained BERT/RoBERTa/T5 models), use
  Hugging Face Spaces instead (root `Dockerfile`, more RAM).

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
