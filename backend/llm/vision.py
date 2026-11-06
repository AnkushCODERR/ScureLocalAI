from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from backend.config import Settings, VisionSettings, get_settings


logger = logging.getLogger(__name__)


def encode_image(image_input: bytes | str | Path, max_dimension: int = 1024) -> str:
    """Read an image, resize if excessively large, and return a base64 encoded string.

    Supports:
    - Raw image bytes
    - Path object or filesystem path string
    - Data URL string (e.g. data:image/jpeg;base64,...)
    - Raw base64 string
    """
    raw_bytes: bytes

    if isinstance(image_input, bytes):
        raw_bytes = image_input
    elif isinstance(image_input, Path):
        if not image_input.is_file():
            raise FileNotFoundError(f"Image not found at {image_input}")
        raw_bytes = image_input.read_bytes()
    elif isinstance(image_input, str):
        # 1. Handle data URL (data:image/...;base64,...)
        if image_input.startswith("data:") and "," in image_input:
            _, b64_part = image_input.split(",", 1)
            try:
                raw_bytes = base64.b64decode(b64_part)
            except Exception as exc:
                raise ValueError(f"Invalid base64 payload in data URL: {exc}") from exc
        else:
            # 2. Check if it is a real file on disk
            is_file = False
            try:
                p = Path(image_input)
                if p.is_file():
                    is_file = True
                    raw_bytes = p.read_bytes()
            except Exception:
                is_file = False

            if not is_file:
                # 3. Try decoding as raw base64 string
                try:
                    raw_bytes = base64.b64decode(image_input)
                    if not raw_bytes:
                        raise ValueError("Decoded base64 is empty")
                except Exception as exc:
                    raise FileNotFoundError(
                        f"Image not found at {image_input} and could not be decoded as base64: {exc}"
                    ) from exc
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img = img.convert("RGB")
            width, height = img.size
            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=88)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.warning("Failed to optimize image with PIL (%s), using raw bytes", exc)
        return base64.b64encode(raw_bytes).decode("utf-8")


def _heuristic_figure_fallback(
    kind: str,
    caption: str | None = None,
    page_context: str | None = None,
    width_height: tuple[int, int] | None = None,
) -> str:
    """Generate a clean visual fallback description if the local VLM is offline or busy."""
    kind_labels = {
        "chart": "Chart or graphical data visualization",
        "diagram_or_illustration": "Diagram, schematic, or process flowchart",
        "photo": "Photograph or continuous-tone picture",
        "unknown": "Visual figure or document illustration",
    }
    label = kind_labels.get(kind, "Visual illustration")
    parts = [f"This visual element is classified as a {label}."]
    if caption:
        parts.append(f"Document caption: {caption}.")
    else:
        parts.append("Note: There is no caption provided in the document for this figure.")
    if page_context:
        snippet = " ".join(page_context.split()[:50])
        parts.append(f"Surrounding page context discusses: {snippet}...")
    return " ".join(parts)


async def call_vlm(
    image_input: bytes | str | Path,
    prompt: str,
    settings: VisionSettings | None = None,
    timeout: float = 90.0,
) -> str:
    """Call the local Ollama multimodal model with an image and prompt."""
    resolved = settings or get_settings().vision
    if not resolved.enabled:
        return "Visual language model is disabled in configuration."

    base64_data = encode_image(image_input)
    payload = {
        "model": resolved.model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64_data],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{resolved.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = str(data.get("message", {}).get("content", "")).strip()
            if content:
                return content
    except Exception as exc:
        logger.warning("Ollama VLM call to model '%s' failed: %s", resolved.model, exc)

    return ""


async def describe_document_figure(
    image_path: Path,
    caption: str | None = None,
    kind: str = "unknown",
    page_context: str | None = None,
    settings: VisionSettings | None = None,
) -> str:
    """Generate a rich, grounded description of a document figure.

    Explains the image even if no description or caption exists in the document.
    """
    prompt = (
        "You are an expert technical visual analyst. Examine this visual element extracted from a document page.\n"
        f"Coarse classification: {kind}\n"
        f"Document Caption: {caption if caption else 'None (no description exists in the document)'}\n"
        f"Surrounding Document Text: {page_context if page_context else 'None'}\n\n"
        "Provide a clear, detailed, and self-contained explanation of this image:\n"
        "1. Identify what the image depicts (e.g., bar chart, line graph, scatter plot, system architecture, flowchart, photograph, schematic, table layout).\n"
        "2. Detail the visual content: identify axes, metrics, trends, key data points, arrows, boxes, stages, or components.\n"
        "3. Transcribe and interpret any embedded labels, numbers, or text visible in the image.\n"
        "4. Summarize the key takeaway or significance of this visual.\n"
        "Be specific, informative, and objective."
    )

    description = await call_vlm(image_path, prompt, settings=settings, timeout=60.0)
    if not description:
        description = _heuristic_figure_fallback(kind, caption, page_context)

    return description


async def explain_image_detail(
    image_input: bytes | str | Path,
    user_prompt: str | None = None,
    doc_context: str | None = None,
    settings: VisionSettings | None = None,
) -> str:
    """Provide a standalone detailed visual explanation of any image."""
    base_prompt = user_prompt.strip() if user_prompt else (
        "Explain this image in comprehensive detail. Describe all objects, charts, diagrams, "
        "labels, relationships, data trends, and text shown in the image. State the core concept "
        "or takeaway clearly."
    )
    if doc_context:
        prompt = f"Document context:\n{doc_context}\n\nUser request:\n{base_prompt}"
    else:
        prompt = base_prompt

    result = await call_vlm(image_input, prompt, settings=settings, timeout=90.0)
    if not result:
        return "Could not generate an AI explanation for this image. Please ensure Ollama is running with a vision model like gemma3:4b."
    return result


async def check_image_relevance(
    image_input: bytes | str | Path,
    user_query: str | None,
    candidate_chunks: list[dict[str, Any]],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Evaluate whether an external image is relevant to indexed documents.

    1. VLM extracts visual details, entities, and concepts from the image.
    2. Compares against retrieved document excerpts and figures.
    3. Evaluates relevance degree (High, Moderate, Not Relevant), score (0-100), and rationale.
    """
    resolved_settings = settings or get_settings()

    # Step 1: Get image semantic description from VLM
    vision_prompt = (
        "Analyze this external image thoroughly for document matching:\n"
        "1. Describe all visual contents, charts, diagrams, flowcharts, photographs, or schematics.\n"
        "2. List all visible text, headers, labels, axes, metrics, and variables.\n"
        "3. Summarize the core domain, topic, entities, and purpose of this image."
    )
    image_description = await call_vlm(image_input, vision_prompt, resolved_settings.vision, timeout=60.0)
    if not image_description:
        image_description = "Visual image containing graphics, text, or pictorial elements."

    if not candidate_chunks:
        return {
            "is_relevant": False,
            "relevance_score": 0,
            "relevance_label": "Not Relevant",
            "analysis": "No documents are currently indexed in the knowledge base to compare against.",
            "image_description": image_description,
            "matching_chunks": [],
        }

    # Step 2: Format context for comparison
    excerpts: list[str] = []
    for idx, chunk in enumerate(candidate_chunks[:5], start=1):
        filename = chunk.get("filename", "unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        excerpts.append(f"[{idx}] (Source: {filename}, page {page})\n{text}")

    context_str = "\n\n".join(excerpts)

    # Step 3: Prompt the LLM to assess relevance
    assessment_prompt = (
        "You are an objective document-to-image relevance analyzer.\n\n"
        f"EXTERNAL IMAGE DESCRIPTION:\n{image_description}\n\n"
        f"USER FOCUS / QUESTION:\n{user_query if user_query else 'Is this image relevant to the documents?'}\n\n"
        f"INDEXED DOCUMENT EXCERPTS:\n{context_str}\n\n"
        "TASK:\n"
        "1. Determine whether the external image is relevant to the provided document excerpts.\n"
        "2. State RELEVANCE: [Highly Relevant | Moderately Relevant | Not Relevant]\n"
        "3. State SCORE: [a single integer percentage between 0 and 100]\n"
        "4. Provide a clear RATIONALE explaining:\n"
        "   - Which specific concepts, data points, or topics in the image match the documents.\n"
        "   - Any differences or domain mismatches.\n"
        "   - Specific documents and page numbers that correspond.\n"
    )

    payload = {
        "model": resolved_settings.llm.model,
        "messages": [
            {"role": "system", "content": "You are a precise multimodal document relevance evaluator. Always provide RELEVANCE: and SCORE: followed by your detailed rationale."},
            {"role": "user", "content": assessment_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(f"{resolved_settings.llm.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            raw_answer = str(data.get("message", {}).get("content", ""))
    except Exception as exc:
        logger.warning("LLM assessment failed: %s", exc)
        raw_answer = ""

    # Parse relevance label and score
    label = "Moderately Relevant"
    score = 50
    if raw_answer:
        import re

        if re.search(r"highly relevant", raw_answer, re.IGNORECASE):
            label = "Highly Relevant"
            score = 85
        elif re.search(r"not relevant", raw_answer, re.IGNORECASE):
            label = "Not Relevant"
            score = 15
        elif re.search(r"moderately relevant", raw_answer, re.IGNORECASE):
            label = "Moderately Relevant"
            score = 60

        score_match = re.search(r"SCORE:\s*(\d{1,3})%?", raw_answer, re.IGNORECASE)
        if score_match:
            try:
                score = max(0, min(100, int(score_match.group(1))))
                if score >= 70:
                    label = "Highly Relevant"
                elif score >= 35:
                    label = "Moderately Relevant"
                else:
                    label = "Not Relevant"
            except ValueError:
                pass

        analysis = raw_answer
    else:
        # Fallback based on top distance
        top_dist = candidate_chunks[0].get("distance", 1.0)
        similarity = max(0.0, 1.0 - float(top_dist))
        score = int(similarity * 100)
        if score >= 70:
            label = "Highly Relevant"
        elif score >= 35:
            label = "Moderately Relevant"
        else:
            label = "Not Relevant"
        analysis = (
            f"Based on semantic embedding similarity ({score}%), the external image appears {label.lower()} "
            f"to {candidate_chunks[0].get('filename', 'the document')}."
        )

    is_relevant = score >= 35

    matching_chunks = [
        {
            "filename": chunk.get("filename", ""),
            "page": chunk.get("page"),
            "text": chunk.get("text", "")[:300] + ("..." if len(chunk.get("text", "")) > 300 else ""),
            "distance": round(chunk.get("distance", 0.0), 3),
            "is_image": bool(chunk.get("is_image", False)),
            "image_url": chunk.get("image_url", ""),
            "caption": chunk.get("caption", ""),
        }
        for chunk in candidate_chunks[:4]
    ]

    return {
        "is_relevant": is_relevant,
        "relevance_score": score,
        "relevance_label": label,
        "analysis": analysis,
        "image_description": image_description,
        "matching_chunks": matching_chunks,
    }
