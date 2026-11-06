from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from backend.config import Settings, get_settings
from backend.llm.vision import check_image_relevance, explain_image_detail
from backend.retrieval.embedder import embed
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/vision", tags=["vision"])


class ExplainResponse(BaseModel):
    explanation: str
    filename: str = ""


class RelevanceResponse(BaseModel):
    is_relevant: bool
    relevance_score: int
    relevance_label: str
    analysis: str
    image_description: str
    matching_chunks: list[dict[str, Any]]


@router.post("/explain", response_model=ExplainResponse)
async def explain_image(
    file: UploadFile = File(...),
    prompt: str = Form(None),
    doc_context: str = Form(None),
    settings: Settings = Depends(get_settings),
) -> ExplainResponse:
    """Provide a comprehensive explanation of an uploaded image even if it has no description."""
    try:
        content = await file.read()
        explanation = await explain_image_detail(
            content,
            user_prompt=prompt,
            doc_context=doc_context,
            settings=settings.vision,
        )
        return ExplainResponse(explanation=explanation, filename=file.filename or "")
    except Exception as exc:
        logger.exception("Failed to explain image: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain image: {exc}",
        ) from exc
    finally:
        await file.close()


@router.post("/check-relevance", response_model=RelevanceResponse)
async def check_external_image_relevance(
    file: UploadFile = File(...),
    query: str = Form(None),
    settings: Settings = Depends(get_settings),
    store: ChromaStore = Depends(get_store),
) -> RelevanceResponse:
    """Analyze an external image and determine its relevance to indexed documents."""
    try:
        content = await file.read()

        # Generate a quick semantic embedding query from the image's filename/query
        search_terms = query.strip() if query else "diagram chart data architecture findings results"
        embedding = (await embed([search_terms], settings.embeddings))[0]

        # Retrieve top candidates across documents and figures
        candidate_chunks = store.query(embedding, k=8)

        # Run the detailed relevance assessment
        result = await check_image_relevance(
            image_input=content,
            user_query=query,
            candidate_chunks=candidate_chunks,
            settings=settings,
        )
        return RelevanceResponse(**result)
    except Exception as exc:
        logger.exception("Failed to check image relevance: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check image relevance: {exc}",
        ) from exc
    finally:
        await file.close()
