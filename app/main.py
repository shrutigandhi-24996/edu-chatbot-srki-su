"""FastAPI application for the Educational Chatbot."""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import INSTITUTIONS, settings
from app.pipeline import web_scraper
from app.pipeline.orchestrator import Orchestrator
from app.schemas import ChatRequest, ChatResponse, HealthResponse

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="Educational Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()


def _frontend_file(name: str) -> Path:
    path = FRONTEND_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Missing frontend asset: {name}")
    return path


@app.on_event("startup")
def _warmup() -> None:
    """Build the active institution's assistant in the background."""
    if FRONTEND_DIR.is_dir():
        print(f"[ui] frontend ready at {FRONTEND_DIR}")
    else:
        print(f"[ui] WARNING: frontend folder missing at {FRONTEND_DIR}")
    threading.Thread(
        target=lambda: orchestrator.get(settings.active_institution), daemon=True
    ).start()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    inst = settings.institution(settings.active_institution)
    if not orchestrator.is_built(inst.code):
        # Still warming up — respond fast with 200 so health checks pass.
        return HealthResponse(
            status="warming",
            active_institution=inst.name,
            intent_backend="loading",
            generator_ready=False,
            web_cache_pages=len(web_scraper.load_cache(inst.code)),
            labels=[],
        )
    assistant = orchestrator.get(inst.code)
    return HealthResponse(
        status="ok",
        active_institution=assistant.inst.name,
        intent_backend=assistant.intent.backend,
        generator_ready=assistant.generator.ready,
        llm_brain=assistant.brain.ready,
        web_cache_pages=len(web_scraper.load_cache(assistant.inst.code)),
        labels=assistant.intent.labels,
    )


@app.get("/api/institutions")
def institutions() -> dict:
    return {
        "active": settings.active_institution,
        "available": [
            {"code": c, "name": i.name, "full_name": i.full_name, "website": i.website}
            for c, i in INSTITUTIONS.items()
        ],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    result = orchestrator.chat(req.message, institution=req.institution)
    return ChatResponse(**result)


@app.post("/api/web/refresh")
def refresh_web(institution: str | None = None) -> dict:
    inst = settings.institution(institution)
    count = web_scraper.refresh_cache(inst, force=True)
    orchestrator.get(inst.code).retriever._load()
    return {"institution": inst.name, "pages_cached": count}


# --- Chat UI (served explicitly so Render always shows the interface) ------
@app.get("/", include_in_schema=False)
@app.get("/chat", include_in_schema=False)
def chat_ui() -> FileResponse:
    return FileResponse(_frontend_file("index.html"), media_type="text/html")


@app.get("/styles.css", include_in_schema=False)
def chat_styles() -> FileResponse:
    return FileResponse(_frontend_file("styles.css"), media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def chat_script() -> FileResponse:
    return FileResponse(_frontend_file("app.js"), media_type="application/javascript")


@app.get("/docs-ui", include_in_schema=False)
def docs_ui_redirect() -> RedirectResponse:
    """Some hosts open /docs by default — send users to the chat UI."""
    return RedirectResponse(url="/", status_code=302)


# Legacy /static/* paths (older index.html references)
if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
