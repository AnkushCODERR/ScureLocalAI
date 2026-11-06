from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.config import ROOT_DIR, ensure_storage_dirs, get_settings
from backend.retrieval.store import get_store
from backend.routers.documents import router as documents_router
from backend.routers.figures import router as figures_router
from backend.routers.query import router as query_router
from backend.routers.vision import router as vision_router
from backend.routers.voice import router as voice_router


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ScureLocalAI - Multimodal & Voice Assistant", version="0.2.0")

app.include_router(documents_router, prefix="/api")
app.include_router(figures_router, prefix="/api")
app.include_router(query_router, prefix="/api")
app.include_router(vision_router, prefix="/api")
app.include_router(voice_router, prefix="/api")


@app.on_event("startup")
async def startup() -> None:
    settings = get_settings()
    ensure_storage_dirs(settings)
    get_store()

    logger.info(
        "ScureLocalAI started with text model '%s' and vision model '%s'",
        settings.llm.model,
        settings.vision.model,
    )


frontend_dir = ROOT_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=Path(frontend_dir), html=True), name="frontend")
