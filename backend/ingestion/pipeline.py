from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from backend.config import Settings, get_settings
from backend.ingestion.chunker import chunk_text
from backend.ingestion.parsers import parse_docx, parse_pdf, parse_pptx, parse_text
from backend.ingestion.Visualizing import (
    VisionConfig,
    detect_figures,
    save_outputs,
)
from backend.llm.vision import describe_document_figure
from backend.retrieval.embedder import embed
from backend.retrieval.store import ChromaStore, get_store


logger = logging.getLogger(__name__)

PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".pptx": parse_pptx,
    ".txt": parse_text,
    ".md": parse_text,
}


def _extract_docx_figures(docx_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Extract embedded images from DOCX documents."""
    from docx import Document

    figures: list[dict[str, Any]] = []
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = Document(str(docx_path))
        idx = 1
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                img_part = rel.target_part
                fig_id = f"docx_img_{idx:02d}"
                crop_path = crops_dir / f"{fig_id}.png"
                try:
                    with Image.open(io.BytesIO(img_part.blob)) as img:
                        img.convert("RGB").save(crop_path, format="PNG")
                    figures.append(
                        {
                            "figure_id": fig_id,
                            "crop_path": str(crop_path),
                            "page": 1,
                            "kind": "illustration",
                            "caption": f"Embedded figure {idx}",
                        }
                    )
                    idx += 1
                except Exception as exc:
                    logger.warning("Failed to save docx image: %s", exc)
    except Exception as exc:
        logger.warning("DOCX image extraction failed: %s", exc)

    return figures


def _extract_pptx_figures(pptx_path: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Extract embedded images from PPTX presentations."""
    from pptx import Presentation

    figures: list[dict[str, Any]] = []
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    try:
        prs = Presentation(str(pptx_path))
        idx = 1
        for slide_index, slide in enumerate(prs.slides, start=1):
            for shape in slide.shapes:
                if getattr(shape, "shape_type", None) == 13:  # Picture
                    fig_id = f"slide{slide_index:02d}_img{idx:02d}"
                    crop_path = crops_dir / f"{fig_id}.png"
                    try:
                        with Image.open(io.BytesIO(shape.image.blob)) as img:
                            img.convert("RGB").save(crop_path, format="PNG")
                        figures.append(
                            {
                                "figure_id": fig_id,
                                "crop_path": str(crop_path),
                                "page": slide_index,
                                "kind": "diagram_or_illustration",
                                "caption": shape.name or f"Slide {slide_index} graphic",
                            }
                        )
                        idx += 1
                    except Exception as exc:
                        logger.warning("Failed to save pptx image: %s", exc)
    except Exception as exc:
        logger.warning("PPTX image extraction failed: %s", exc)

    return figures


async def _process_pdf_figures(
    file_path: Path,
    doc_id: str,
    pages: list[str],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Detect, crop, and generate AI explanations for figures in a PDF."""
    vision_cfg = VisionConfig(
        output_dir=settings.vision.output_dir / doc_id,
        crop_dpi=settings.vision.crop_dpi,
        min_side_pt=settings.vision.min_side_pt,
        cluster_gap_pt=settings.vision.cluster_gap_pt,
        vector_min_score=settings.vision.vector_min_score,
        drop_recurring=settings.vision.drop_recurring,
        raster_enabled=settings.vision.raster_enabled,
    )

    doc_figures_dir = settings.vision.output_dir / doc_id
    doc_figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        detection_result = detect_figures(file_path, cfg=vision_cfg)
        save_outputs(file_path, detection_result, doc_figures_dir, vision_cfg, images=True)
    except Exception as exc:
        logger.warning("PDF figure detection failed for %s: %s", file_path, exc)
        return []

    figure_chunks: list[dict[str, Any]] = []

    for fig in detection_result.figures:
        crop_file = doc_figures_dir / "crops" / f"{fig.figure_id}.png"
        if not crop_file.exists():
            continue

        # Extract surrounding context text from the same page
        page_idx = fig.page - 1
        page_text = pages[page_idx] if 0 <= page_idx < len(pages) else ""
        surrounding = " ".join(page_text.split()[:80])

        # Generate VLM visual explanation (or heuristic fallback)
        explanation = await describe_document_figure(
            image_path=crop_file,
            caption=fig.caption,
            kind=fig.kind,
            page_context=surrounding,
            settings=settings.vision,
        )

        chunk_text_repr = (
            f"[FIGURE / IMAGE: {fig.caption or fig.kind.upper()} on Page {fig.page}]\n"
            f"Classification: {fig.kind}\n"
            f"Caption: {fig.caption if fig.caption else 'None provided in document'}\n"
            f"Visual Explanation: {explanation}\n"
            f"Linked Document Context: {surrounding}"
        )

        figure_chunks.append(
            {
                "text": chunk_text_repr,
                "page": fig.page,
                "chunk_index": 0,  # Will be indexed sequentially
                "is_image": True,
                "figure_id": fig.figure_id,
                "image_url": f"/api/figures/{doc_id}/crops/{fig.figure_id}.png",
                "caption": fig.caption or "",
                "kind": fig.kind,
                "vlm_description": explanation,
                "surrounding_text": surrounding,
            }
        )

    return figure_chunks


async def ingest_document(
    file_path: Path,
    doc_id: str,
    filename: str,
    settings: Settings | None = None,
    store: ChromaStore | None = None,
) -> int:
    resolved = settings or get_settings()
    suffix = file_path.suffix.lower()
    parser = PARSERS.get(suffix)
    if parser is None:
        raise ValueError(f"Unsupported file type: {suffix}")

    # 1. Parse text pages
    pages = parser(file_path)
    text_chunks_raw = chunk_text(
        pages,
        chunk_size=resolved.retrieval.chunk_size,
        overlap=resolved.retrieval.chunk_overlap,
    )

    all_chunks: list[dict[str, Any]] = [
        {
            "text": chunk["text"],
            "page": chunk["page"],
            "chunk_index": idx,
            "is_image": False,
            "figure_id": "",
            "image_url": "",
            "caption": "",
            "kind": "text",
            "vlm_description": "",
            "surrounding_text": "",
        }
        for idx, chunk in enumerate(text_chunks_raw)
    ]

    # 2. Multimodal extraction & visual comprehension
    if resolved.vision.enabled:
        doc_figures_dir = resolved.vision.output_dir / doc_id
        doc_figures_dir.mkdir(parents=True, exist_ok=True)

        figure_chunks: list[dict[str, Any]] = []
        if suffix == ".pdf":
            figure_chunks = await _process_pdf_figures(file_path, doc_id, pages, resolved)
        elif suffix == ".docx":
            raw_figs = _extract_docx_figures(file_path, doc_figures_dir)
            for fig in raw_figs:
                crop_path = Path(fig["crop_path"])
                explanation = await describe_document_figure(
                    crop_path, caption=fig["caption"], kind=fig["kind"], settings=resolved.vision
                )
                figure_chunks.append(
                    {
                        "text": f"[FIGURE: {fig['caption']}]\nVisual Explanation: {explanation}",
                        "page": 1,
                        "chunk_index": 0,
                        "is_image": True,
                        "figure_id": fig["figure_id"],
                        "image_url": f"/api/figures/{doc_id}/crops/{fig['figure_id']}.png",
                        "caption": fig["caption"],
                        "kind": fig["kind"],
                        "vlm_description": explanation,
                        "surrounding_text": "",
                    }
                )
        elif suffix == ".pptx":
            raw_figs = _extract_pptx_figures(file_path, doc_figures_dir)
            for fig in raw_figs:
                crop_path = Path(fig["crop_path"])
                slide_text = pages[fig["page"] - 1] if fig["page"] - 1 < len(pages) else ""
                explanation = await describe_document_figure(
                    crop_path, caption=fig["caption"], kind=fig["kind"], page_context=slide_text, settings=resolved.vision
                )
                figure_chunks.append(
                    {
                        "text": f"[FIGURE: Slide {fig['page']} - {fig['caption']}]\nVisual Explanation: {explanation}\nSlide Content: {slide_text[:100]}",
                        "page": fig["page"],
                        "chunk_index": 0,
                        "is_image": True,
                        "figure_id": fig["figure_id"],
                        "image_url": f"/api/figures/{doc_id}/crops/{fig['figure_id']}.png",
                        "caption": fig["caption"],
                        "kind": fig["kind"],
                        "vlm_description": explanation,
                        "surrounding_text": slide_text,
                    }
                )

        # Assign unique chunk indices
        start_idx = len(all_chunks)
        for offset, fig_chunk in enumerate(figure_chunks):
            fig_chunk["chunk_index"] = start_idx + offset
            all_chunks.append(fig_chunk)

    # 3. Generate embeddings
    texts_to_embed = [str(chunk["text"]) for chunk in all_chunks]
    embeddings = await embed(texts_to_embed, resolved.embeddings)

    # 4. Store in ChromaDB
    (store or get_store()).add_chunks(doc_id, filename, all_chunks, embeddings)
    return len(all_chunks)
