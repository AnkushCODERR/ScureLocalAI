from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import Settings, get_settings
from backend.llm.chat import answer_query, answer_query_stream
from backend.llm.vision import call_vlm
from backend.retrieval.embedder import embed
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class FigureInfo(BaseModel):
    figure_id: str
    filename: str
    page: int
    image_url: str
    caption: str
    kind: str
    vlm_description: str
    surrounding_text: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int | None = Field(default=None, ge=1, le=20)
    image: str | None = Field(default=None, description="Optional base64 encoded image to accompany question")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    figures: list[FigureInfo] = []


def _sources(chunks: list[dict[str, object]]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        filename = str(chunk.get("filename", "")).strip()
        page = chunk.get("page")
        label = f"{filename} (page {page})" if page else filename
        if label and label not in sources:
            sources.append(label)
    return sources


def _extract_figures(chunks: list[dict[str, Any]]) -> list[FigureInfo]:
    figures: list[FigureInfo] = []
    seen: set[str] = set()

    for chunk in chunks:
        if chunk.get("is_image") and chunk.get("image_url"):
            fig_id = str(chunk.get("figure_id", ""))
            key = f"{chunk.get('doc_id')}:{fig_id}"
            if key in seen:
                continue
            seen.add(key)

            figures.append(
                FigureInfo(
                    figure_id=fig_id,
                    filename=str(chunk.get("filename", "")),
                    page=int(chunk.get("page", 1) or 1),
                    image_url=str(chunk.get("image_url", "")),
                    caption=str(chunk.get("caption", "")),
                    kind=str(chunk.get("kind", "unknown")),
                    vlm_description=str(chunk.get("vlm_description", "")),
                    surrounding_text=str(chunk.get("surrounding_text", "")),
                )
            )

    return figures


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")

    try:
        attached_analysis = ""
        if request.image and settings.vision.enabled:
            analysis_prompt = (
                f"User asked: '{question}'. Analyze this attached image in detail. "
                "Describe what is shown, data points, charts, or diagrams."
            )
            attached_analysis = await call_vlm(request.image, analysis_prompt, settings.vision, timeout=60.0)

        search_query = f"{question} {attached_analysis}".strip()
        question_embedding = (await embed([search_query], settings.embeddings))[0]

        top_k = request.k or settings.retrieval.top_k
        chunks = store.query(question_embedding, top_k)

        context_chunks = list(chunks)
        if attached_analysis:
            context_chunks.insert(
                0,
                {
                    "text": f"[USER ATTACHED IMAGE ANALYSIS]\n{attached_analysis}",
                    "filename": "User Attached Image",
                    "page": 1,
                    "is_image": True,
                },
            )

        answer = await answer_query(question, context_chunks, settings.llm)
        figures = _extract_figures(chunks)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to answer query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer query: {exc}",
        ) from exc

    return QueryResponse(answer=answer, sources=_sources(chunks), figures=figures)


@router.post("/stream")
async def query_knowledge_base_stream(
    request: QueryRequest,
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty")

    try:
        attached_analysis = ""
        if request.image and settings.vision.enabled:
            analysis_prompt = (
                f"User asked: '{question}'. Analyze this attached image in detail. "
                "Describe what is shown, data points, charts, or diagrams."
            )
            attached_analysis = await call_vlm(request.image, analysis_prompt, settings.vision, timeout=60.0)

        search_query = f"{question} {attached_analysis}".strip()
        question_embedding = (await embed([search_query], settings.embeddings))[0]

        top_k = request.k or settings.retrieval.top_k
        chunks = store.query(question_embedding, top_k)

        context_chunks = list(chunks)
        if attached_analysis:
            context_chunks.insert(
                0,
                {
                    "text": f"[USER ATTACHED IMAGE ANALYSIS]\n{attached_analysis}",
                    "filename": "User Attached Image",
                    "page": 1,
                    "is_image": True,
                },
            )

        sources = _sources(chunks)
        figures = _extract_figures(chunks)
        figures_dicts = [
            {
                "figure_id": f.figure_id,
                "filename": f.filename,
                "page": f.page,
                "image_url": f.image_url,
                "caption": f.caption,
                "kind": f.kind,
                "vlm_description": f.vlm_description,
                "surrounding_text": f.surrounding_text,
            }
            for f in figures
        ]

        async def event_generator():
            # Send initial metadata (sources & figures) immediately so UI can display them
            yield json.dumps({"type": "init", "sources": sources, "figures": figures_dicts}) + "\n"

            # Stream answer tokens as Ollama produces them
            full_text = ""
            async for token in answer_query_stream(question, context_chunks, settings.llm):
                full_text += token
                yield json.dumps({"type": "token", "token": token}) + "\n"

            # Send done event
            yield json.dumps({"type": "done", "sources": sources, "full_answer": full_text}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    except Exception as exc:
        logger.exception("Failed to initialize stream query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer query: {exc}",
        ) from exc
