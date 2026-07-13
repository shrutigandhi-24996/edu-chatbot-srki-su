"""FastAPI application for the Educational Chatbot."""
from __future__ import annotations

import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import INSTITUTIONS, settings
from app.pipeline import web_scraper
from app.pipeline.orchestrator import Orchestrator
from app.schemas import ChatRequest, ChatResponse, HealthResponse

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = FastAPI(title="Educational Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()


@app.on_event("startup")
def _warmup() -> None:
    """Build the active institution's assistant in the background so the first
    real request (and health checks) are fast and don't time out on deploy."""
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


# --- Static frontend ------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(FRONTEND_DIR / "index.html"))
