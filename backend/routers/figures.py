from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from backend.config import Settings, get_settings
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/figures", tags=["figures"])


@router.get("")
async def list_all_figures(store: ChromaStore = Depends(get_store)) -> list[dict[str, Any]]:
    """List all extracted figures across all documents."""
    return store.list_figures()


@router.get("/{doc_id}/crops/{filename}")
async def get_figure_crop(
    doc_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve an extracted figure crop image."""
    safe_doc = Path(doc_id).name
    safe_file = Path(filename).name
    crop_path = settings.vision.output_dir / safe_doc / "crops" / safe_file
    if not crop_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Figure crop '{filename}' not found for document {doc_id}",
        )
    return FileResponse(crop_path, media_type="image/png")


@router.get("/{doc_id}/{filename}")
async def get_figure_image(
    doc_id: str,
    filename: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Serve an extracted full figure or annotated page image."""
    safe_doc = Path(doc_id).name
    safe_file = Path(filename).name

    # Check crops subfolder first
    crop_path = settings.vision.output_dir / safe_doc / "crops" / safe_file
    if crop_path.is_file():
        return FileResponse(crop_path, media_type="image/png")

    # Check document figure root directory
    file_path = settings.vision.output_dir / safe_doc / safe_file
    if not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Figure image '{filename}' not found for document {doc_id}",
        )
    return FileResponse(file_path, media_type="image/png")


@router.get("/{doc_id}")
async def list_document_figures(
    doc_id: str,
    store: ChromaStore = Depends(get_store),
) -> list[dict[str, Any]]:
    """List figures for a specific document."""
    return store.list_figures(doc_id)
